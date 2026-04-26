# src/core/sql_agent_store.py
"""
SQL Agent 存储包装器，接口与 VikingStoreWrapper / PageIndexStoreWrapper / HippoRAGStoreWrapper 对齐。

内部使用 LangChain SQL Agent + SQLite 实现文档入库与检索。
每个数据集有独立的 SQL schema 和入库逻辑，与 LangChain-SQL-Agent 项目保持一致。
"""

import csv
import json
import os
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from sqlalchemy import create_engine, event, text

from src.adapters.base import StandardDoc
from src.core.logger import get_logger

logger = get_logger()

CHUNK_SIZE_DEFAULT = 800
CHUNK_OVERLAP_DEFAULT = 100
CHUNK_MIN_SIZE_DEFAULT = 200
CHUNK_MAX_SIZE_DEFAULT = 1200

_SENT_DELIM_RE = re.compile(r'([.。!！?？;；\n]\s*)')


@dataclass
class SQLAgentResource:
    uri: str
    content: str = ""
    score: float = 0.0


@dataclass
class SQLAgentResult:
    resources: List[SQLAgentResource] = field(default_factory=list)
    retrieve_input_tokens: int = 0
    retrieve_output_tokens: int = 0
    agent_answer: str = ""
    sql_queries: List[str] = field(default_factory=list)


def _chunk_text(text_content: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """简单字符级滑动窗口分片，与 LangChain-SQL-Agent 保持一致"""
    if not text_content:
        return []
    chunks = []
    start = 0
    while start < len(text_content):
        end = start + chunk_size
        chunks.append(text_content[start:end])
        start = end - chunk_overlap
    return chunks


def _build_overlap(chunk_text: str, overlap_size: int) -> str:
    """从分片尾部提取 overlap 区域，尽量对齐到句子边界"""
    if overlap_size <= 0 or not chunk_text:
        return ""
    tail = chunk_text[-overlap_size:]
    m = re.search(r'[.。!！?？;；\n]\s*', tail)
    if m:
        return tail[m.end():]
    return tail


def _chunk_text_sentence_aware(
    text_content: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    chunk_min_size: int = 200,
    chunk_max_size: int = 1200,
) -> List[str]:
    """句子感知分片：不切断完整句子，合并过小分片"""
    if not text_content:
        return []
    chunk_max_size = max(chunk_max_size, chunk_size)

    # 用捕获组分割，保留分隔符（标点+空白），再拼回句子
    parts = _SENT_DELIM_RE.split(text_content)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i] + parts[i + 1]
        if s:
            sentences.append(s)
    if len(parts) % 2 == 1 and parts[-1]:
        sentences.append(parts[-1])
    if not sentences:
        return [text_content]

    chunks = []
    current_chunk = ""
    sent_idx = 0

    while sent_idx < len(sentences):
        sent = sentences[sent_idx]
        tentative_len = len(current_chunk) + len(sent)

        if not current_chunk:
            current_chunk = sent
            sent_idx += 1
            continue

        if tentative_len <= chunk_size:
            current_chunk += sent
            sent_idx += 1
            continue

        if tentative_len <= chunk_max_size:
            current_chunk += sent
            sent_idx += 1

        # 封闭当前分片
        chunks.append(current_chunk)
        overlap_prefix = _build_overlap(current_chunk, chunk_overlap)
        current_chunk = overlap_prefix

    # 处理最后一个分片
    if current_chunk:
        if len(current_chunk) < chunk_min_size and chunks:
            chunks[-1] += current_chunk
        else:
            chunks.append(current_chunk)

    # 后处理：合并过小分片
    final = []
    for chunk in chunks:
        if final and len(chunk) < chunk_min_size:
            final[-1] += chunk
        else:
            final.append(chunk)
    return final


class SQLAgentStoreWrapper:
    """SQL Agent 存储包装器。按 dataset_name 分发到不同的 schema/ingest 逻辑。"""

    def __init__(self, store_path: str, sql_agent_config: Optional[dict] = None):
        self.store_path = store_path
        self.logger = logger
        os.makedirs(store_path, exist_ok=True)

        cfg = sql_agent_config or {}
        self.db_path = os.path.join(store_path, cfg.get('db_name', 'docs.db'))
        self.dataset_name = cfg.get('dataset_name', '').lower()
        self.raw_data_path = cfg.get('raw_data_path', '')
        self.chunk_size = int(cfg.get('chunk_size', CHUNK_SIZE_DEFAULT))
        self.chunk_overlap = int(cfg.get('chunk_overlap', CHUNK_OVERLAP_DEFAULT))
        self.sentence_aware_chunk = cfg.get('sentence_aware_chunk', False)
        self.chunk_min_size = int(cfg.get('chunk_min_size', CHUNK_MIN_SIZE_DEFAULT))
        self.chunk_max_size = int(cfg.get('chunk_max_size', CHUNK_MAX_SIZE_DEFAULT))
        self.max_iterations = int(cfg.get('max_iterations', 15))
        self.verbose = cfg.get('verbose', False)

        # LLM 配置
        self.llm_model = cfg.get('llm_model', '')
        self.llm_base_url = cfg.get('llm_base_url', '')
        self.llm_api_key = cfg.get('llm_api_key', '')
        self.llm_api_key_env = cfg.get('llm_api_key_env', '')

        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"tiktoken init failed, token counting will return 0: {e}")
            self.enc = None

        self._engine = None
        self._engine_lock = threading.Lock()
        self._token_tracker = None
        self._agent_executor = None

    # ---- engine / agent 基础设施 ----

    def _do_chunk(self, content: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """统一分片入口，根据 sentence_aware_chunk 配置选择策略"""
        cs = chunk_size or self.chunk_size
        co = chunk_overlap or self.chunk_overlap
        if self.sentence_aware_chunk:
            return _chunk_text_sentence_aware(content, cs, co,
                                              self.chunk_min_size, self.chunk_max_size)
        return _chunk_text(content, cs, co)

    def _get_engine(self):
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    self._engine = create_engine(
                        f"sqlite:///{self.db_path}",
                        connect_args={"timeout": 30},
                    )
                    @event.listens_for(self._engine, "connect")
                    def _set_pragmas(dbapi_conn, _rec):
                        cur = dbapi_conn.cursor()
                        cur.execute("PRAGMA journal_mode=WAL")
                        cur.execute("PRAGMA busy_timeout=10000")
                        cur.close()
        return self._engine

    def _get_api_key(self) -> str:
        if self.llm_api_key:
            return self.llm_api_key
        if self.llm_api_key_env:
            return os.environ.get(self.llm_api_key_env, '')
        return ''

    def _ensure_agent(self):
        if self._agent_executor is not None:
            return self._agent_executor

        from langchain_community.agent_toolkits import SQLDatabaseToolkit
        from langchain_community.utilities import SQLDatabase
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_openai import ChatOpenAI
        try:
            from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
        except ImportError:
            from langchain.agents import AgentExecutor, create_tool_calling_agent
        from src.core.sql_agent_token_tracker import TokenTracker

        engine = self._get_engine()
        db = SQLDatabase(engine=engine, max_string_length=10000)
        self._token_tracker = TokenTracker()
        llm = ChatOpenAI(
            model=self.llm_model, api_key=self._get_api_key(),
            base_url=self.llm_base_url, temperature=0,
            callbacks=[self._token_tracker],
        )
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a precise SQL database retrieval assistant. "
             "Your ONLY job is to find and return relevant source texts from the database. "
             "Use the provided tools to explore the database schema and run queries. "
             "ALWAYS use the tools to get real data — never fabricate results. "
             "Do NOT answer the question yourself. Only return the relevant raw texts from the database. "
             "Return each relevant text on a separate line, prefixed with '- '. "),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)
        self._agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=self.verbose,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True, return_intermediate_steps=True,
        )
        return self._agent_executor

    def count_tokens(self, t: str) -> int:
        if not t or not self.enc:
            return 0
        return len(self.enc.encode(str(t)))

    def _read_document(self, doc_path: str) -> str:
        ext = os.path.splitext(doc_path)[1].lower()
        if ext == '.pdf':
            return self._extract_pdf_text(doc_path)
        with open(doc_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read().strip()

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF: pdfplumber -> pypdf -> docling fallback chain."""
        # Priority 1: pdfplumber
        try:
            import pdfplumber
            self.logger.info("Attempting to extract text using pdfplumber")
            pages_text = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages_text.append(t)
            content = "\n\n".join(pages_text)
            if content.strip():
                return content
        except ImportError:
            pass
        except Exception as exc:
            self.logger.warning("pdfplumber failed for %s: %s", pdf_path, exc)
        # Priority 2: docling
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            content = result.document.export_to_markdown()
            if content.strip():
                return content
        except ImportError:
            pass
        except Exception as exc:
            self.logger.warning("docling failed for %s: %s, falling back", pdf_path, exc)

        # Priority 3: pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            if reader.is_encrypted:
                reader.decrypt("")
            content = ""
            for page in reader.pages:
                content += (page.extract_text() or "") + "\n"
            if content.strip():
                return content
        except ImportError:
            pass
        except Exception as exc:
            self.logger.warning("pypdf failed for %s: %s, falling back", pdf_path, exc)


        self.logger.error(
            "Cannot extract text from %s. "
            "Install one of: pip install 'docling>=2' / pip install pypdf / pip install pdfplumber",
            pdf_path,
        )
        return ""

    # ================================================================
    #  ingest 入口：按 dataset_name 分发
    # ================================================================

    def ingest(self, samples: List[StandardDoc], max_workers: int = 4, monitor=None) -> dict:
        start_time = time.time()
        engine = self._get_engine()

        dispatch = {
            'locomo': self._ingest_locomo,
            'qasper': self._ingest_qasper,
            'hotpotqa': self._ingest_hotpotqa,
            'syllabusqa': self._ingest_syllabusqa,
            'clapnq': self._ingest_clapnq,
            'financebench': self._ingest_financebench,
        }
        handler = dispatch.get(self.dataset_name)
        if handler is None:
            self.logger.warning(f"[SQLAgent] Unknown dataset '{self.dataset_name}', using generic ingest")
            handler = self._ingest_generic

        result = handler(engine, samples)
        if isinstance(result, tuple):
            input_tokens, insert_time = result
        else:
            input_tokens, insert_time = result or 0, time.time() - start_time
        return {"time": time.time() - start_time, "insert_time": insert_time, "input_tokens": input_tokens, "output_tokens": 0}

    # ---- LoCoMo ----

    def _ingest_locomo(self, engine, samples: List[StandardDoc]):
        schema = """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT, session_id INTEGER, session_date TEXT,
            turn_number INTEGER, dia_id TEXT, speaker TEXT, text TEXT
        );
        CREATE TABLE IF NOT EXISTS session_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT, session_id INTEGER, summary TEXT
        );
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT, session_id INTEGER, speaker TEXT,
            observation TEXT, dia_id TEXT
        );"""
        with engine.connect() as conn:
            for stmt in schema.strip().split(';'):
                if stmt.strip():
                    conn.execute(text(stmt))
            conn.commit()

        # 读取原始 JSON，预过滤
        data = self._load_json(self.raw_data_path)
        sample_ids = {s.sample_id for s in samples}
        filtered = [e for e in data if str(e.get('sample_id', '')) in sample_ids] if sample_ids else data
        total_tokens = 0

        # 预处理：提取所有待插入记录并统计 tokens
        conv_rows, summary_rows, obs_rows = [], [], []
        # 每个 sample 的有效 session 数量
        LOCOMO_SESSION_NUM = {
            'conv-26': 19, 'conv-30': 19, 'conv-41': 32, 'conv-42': 29,
            'conv-43': 29, 'conv-44': 28, 'conv-47': 31, 'conv-48': 30,
            'conv-49': 25, 'conv-50': 30,
        }
        for entry in filtered:
            sid = str(entry.get('sample_id', ''))
            conv = entry.get('conversation', {})
            max_sess = LOCOMO_SESSION_NUM.get(sid, 50)
            for sess_idx in range(1, max_sess + 1):
                sess_key = f'session_{sess_idx}'
                if sess_key not in conv or not isinstance(conv[sess_key], list):
                    continue
                date_key = f'{sess_key}_date_time'
                sess_date = conv.get(date_key, '')
                for turn_idx, turn in enumerate(conv[sess_key]):
                    turn_text = turn.get('text', '')
                    total_tokens += self.count_tokens(turn_text)
                    conv_rows.append({"sample_id": sid, "session_id": sess_idx, "session_date": sess_date,
                                      "turn_number": turn_idx + 1, "dia_id": turn.get('dia_id', ''),
                                      "speaker": turn.get('speaker', ''), "text": turn_text})

            summaries = entry.get('session_summary', {})
            for i in range(1, max_sess + 1):
                skey = f'session_{i}_summary'
                if skey in summaries and summaries[skey]:
                    total_tokens += self.count_tokens(summaries[skey])
                    summary_rows.append({"sample_id": sid, "session_id": i, "summary": summaries[skey]})

            observations = entry.get('observation', {})
            for i in range(1, max_sess + 1):
                okey = f'session_{i}_observation'
                obs_dict = observations.get(okey)
                if not obs_dict or not isinstance(obs_dict, dict):
                    continue
                for speaker, obs_list in obs_dict.items():
                    if not isinstance(obs_list, list):
                        continue
                    for obs_item in obs_list:
                        if isinstance(obs_item, list) and len(obs_item) >= 2:
                            dia_ref = obs_item[1]
                            if isinstance(dia_ref, list):
                                dia_ref = ', '.join(str(x) for x in dia_ref)
                            total_tokens += self.count_tokens(str(obs_item[0]))
                            obs_rows.append({"sample_id": sid, "session_id": i, "speaker": speaker,
                                             "observation": str(obs_item[0]), "dia_id": str(dia_ref)})

        # 纯 INSERT 计时
        t_insert = time.time()
        with engine.connect() as conn:
            for r in conv_rows:
                conn.execute(text(
                    "INSERT INTO conversations "
                    "(sample_id,session_id,session_date,turn_number,dia_id,speaker,text) "
                    "VALUES (:sample_id,:session_id,:session_date,:turn_number,:dia_id,:speaker,:text)"
                ), r)
            for r in summary_rows:
                conn.execute(text(
                    "INSERT INTO session_summaries (sample_id,session_id,summary) "
                    "VALUES (:sample_id,:session_id,:summary)"
                ), r)
            for r in obs_rows:
                conn.execute(text(
                    "INSERT INTO observations "
                    "(sample_id,session_id,speaker,observation,dia_id) "
                    "VALUES (:sample_id,:session_id,:speaker,:observation,:dia_id)"
                ), r)
            conn.commit()
        insert_time = time.time() - t_insert
        self.logger.info(f"[SQLAgent-LoCoMo] Ingested for {len(filtered)} samples, {total_tokens} tokens, insert_time={insert_time:.2f}s")
        return total_tokens, insert_time

    # ---- HotpotQA ----

    def _ingest_hotpotqa(self, engine, samples: List[StandardDoc]):
        schema = """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT, title TEXT, content TEXT
        );
        CREATE TABLE IF NOT EXISTS article_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT, title TEXT, chunk_index INTEGER, content TEXT
        );"""
        self._exec_schema(engine, schema)

        data = self._load_json(self.raw_data_path)
        sample_ids = {s.sample_id for s in samples}
        total_chunks = 0
        total_tokens = 0

        # 预处理：提取待插入记录并统计 tokens
        article_rows = []
        chunk_rows = []
        for entry in data:
            sid = str(entry.get('id', ''))
            context = entry.get('context', {})
            titles = context.get('title', [])
            sentences = context.get('sentences', [])

            for i, title in enumerate(titles):
                sents = sentences[i] if i < len(sentences) else []
                content = ' '.join(s.strip() for s in sents if s.strip())
                if not content:
                    continue
                total_tokens += self.count_tokens(content)
                article_rows.append({"sample_id": sid, "title": title, "content": content})
                for ci, chunk in enumerate(self._do_chunk(content)):
                    chunk_rows.append({"sample_id": sid, "title": title, "chunk_index": ci, "content": chunk})
                    total_chunks += 1

        # 纯 INSERT 计时
        t_insert = time.time()
        with engine.connect() as conn:
            for r in article_rows:
                conn.execute(text(
                    "INSERT INTO articles (sample_id,title,content) VALUES (:sample_id,:title,:content)"
                ), r)
            for r in chunk_rows:
                conn.execute(text(
                    "INSERT INTO article_chunks (sample_id,title,chunk_index,content) "
                    "VALUES (:sample_id,:title,:chunk_index,:content)"
                ), r)
            conn.commit()
        insert_time = time.time() - t_insert
        self.logger.info(f"[SQLAgent-HotpotQA] {len(article_rows)} articles, {total_chunks} chunks, {total_tokens} tokens, insert_time={insert_time:.2f}s")
        return total_tokens, insert_time

    # ---- Qasper ----

    def _ingest_qasper(self, engine, samples: List[StandardDoc]):
        schema = """
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT UNIQUE, title TEXT, abstract TEXT
        );
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT, section_index INTEGER, section_name TEXT, content TEXT
        );
        CREATE TABLE IF NOT EXISTS section_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT, section_index INTEGER, chunk_index INTEGER, content TEXT
        );"""
        self._exec_schema(engine, schema)

        # Qasper 可能有多个 split 文件
        papers = self._load_qasper_papers()
        sample_ids = {s.sample_id for s in samples}
        filtered_papers = [p for p in papers if p['id'] in sample_ids] if sample_ids else papers
        total_chunks = 0
        total_tokens = 0

        # 预处理：提取待插入记录并统计 tokens
        paper_rows = []
        section_rows = []
        chunk_rows = []
        for paper in filtered_papers:
            pid = paper['id']
            abstract = paper.get('abstract', '')
            total_tokens += self.count_tokens(abstract)
            paper_rows.append({"paper_id": pid, "title": paper.get('title', ''), "abstract": abstract})

            ft = paper.get('full_text', [])
            sections = []
            if isinstance(ft, list):
                for i, sec in enumerate(ft):
                    sname = sec.get('section_name', '')
                    paras = sec.get('paragraphs', [])
                    content = '\n'.join(paras) if isinstance(paras, list) else str(paras)
                    sections.append((i, sname, content))
            else:
                sec_names = ft.get('section_name', [])
                paragraphs = ft.get('paragraphs', [])
                for i, (sname, paras) in enumerate(zip(sec_names, paragraphs)):
                    content = '\n'.join(paras) if isinstance(paras, list) else str(paras)
                    sections.append((i, sname, content))

            for i, sname, content in sections:
                total_tokens += self.count_tokens(content)
                section_rows.append({"paper_id": pid, "section_index": i, "section_name": sname, "content": content})
                for ci, chunk in enumerate(self._do_chunk(content)):
                    chunk_rows.append({"paper_id": pid, "section_index": i, "chunk_index": ci, "content": chunk})
                    total_chunks += 1

        # 纯 INSERT 计时
        t_insert = time.time()
        with engine.connect() as conn:
            for r in paper_rows:
                conn.execute(text(
                    "INSERT OR IGNORE INTO papers (paper_id,title,abstract) VALUES (:paper_id,:title,:abstract)"
                ), r)
            for r in section_rows:
                conn.execute(text(
                    "INSERT INTO sections (paper_id,section_index,section_name,content) "
                    "VALUES (:paper_id,:section_index,:section_name,:content)"
                ), r)
            for r in chunk_rows:
                conn.execute(text(
                    "INSERT INTO section_chunks (paper_id,section_index,chunk_index,content) "
                    "VALUES (:paper_id,:section_index,:chunk_index,:content)"
                ), r)
            conn.commit()
        insert_time = time.time() - t_insert
        self.logger.info(f"[SQLAgent-Qasper] {len(papers)} papers, {total_chunks} chunks, {total_tokens} tokens, insert_time={insert_time:.2f}s")
        return total_tokens, insert_time

    # ---- SyllabusQA ----

    def _ingest_syllabusqa(self, engine, samples: List[StandardDoc]):
        schema = """
        CREATE TABLE IF NOT EXISTS syllabi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            syllabus_name TEXT, course TEXT, major TEXT, area TEXT,
            university TEXT, num_pages INTEGER, content TEXT
        );
        CREATE TABLE IF NOT EXISTS syllabus_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            syllabus_name TEXT, chunk_index INTEGER, content TEXT
        );"""
        self._exec_schema(engine, schema)

        # 加载元数据 CSV
        meta = {}
        meta_path = os.path.join(self.raw_data_path, 'syllabi_metadata.csv')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    meta[row.get('name', '')] = row

        total_chunks = 0
        total_tokens = 0

        # 预处理：读取文档内容并统计 tokens
        syllabus_rows = []
        chunk_rows = []
        for sample in samples:
            name = sample.sample_id
            content = ''
            for dp in sample.doc_paths:
                content += self._read_document(dp)
            total_tokens += self.count_tokens(content)
            m = meta.get(name, {})
            syllabus_rows.append({"syllabus_name": name, "course": m.get('course', ''), "major": m.get('major', ''),
                                  "area": m.get('area', ''), "university": m.get('university', ''),
                                  "num_pages": int(m.get('num_pages', 0) or 0), "content": content})
            for ci, chunk in enumerate(self._do_chunk(content)):
                chunk_rows.append({"syllabus_name": name, "chunk_index": ci, "content": chunk})
                total_chunks += 1

        # 纯 INSERT 计时
        t_insert = time.time()
        with engine.connect() as conn:
            for r in syllabus_rows:
                conn.execute(text(
                    "INSERT INTO syllabi "
                    "(syllabus_name,course,major,area,university,num_pages,content) "
                    "VALUES (:syllabus_name,:course,:major,:area,:university,:num_pages,:content)"
                ), r)
            for r in chunk_rows:
                conn.execute(text(
                    "INSERT INTO syllabus_chunks (syllabus_name,chunk_index,content) "
                    "VALUES (:syllabus_name,:chunk_index,:content)"
                ), r)
            conn.commit()
        insert_time = time.time() - t_insert
        self.logger.info(f"[SQLAgent-SyllabusQA] {len(samples)} syllabi, {total_chunks} chunks, {total_tokens} tokens, insert_time={insert_time:.2f}s")
        return total_tokens, insert_time

    # ---- ClapNQ ----

    def _ingest_clapnq(self, engine, samples: List[StandardDoc]):
        schema = """
        CREATE TABLE IF NOT EXISTS passages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            qa_id TEXT, title TEXT, content TEXT
        );
        CREATE TABLE IF NOT EXISTS passage_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            qa_id TEXT, title TEXT, chunk_index INTEGER, content TEXT
        );"""
        self._exec_schema(engine, schema)

        # 加载 JSONL 数据，预过滤
        qa_data = self._load_clapnq_data()
        sample_ids = {s.sample_id for s in samples}
        total_chunks = 0
        total_tokens = 0

        # 预处理：提取待插入记录并统计 tokens
        passage_rows = []
        chunk_rows = []
        for item in qa_data:
            qa_id = str(item.get('qa_id', item.get('id', '')))
            if sample_ids and qa_id not in sample_ids:
                continue
            for pg in item.get('passages', []):
                title = pg.get('title', '')
                content = pg.get('text', '')
                total_tokens += self.count_tokens(content)
                passage_rows.append({"qa_id": qa_id, "title": title, "content": content})
                for ci, chunk in enumerate(self._do_chunk(content)):
                    chunk_rows.append({"qa_id": qa_id, "title": title, "chunk_index": ci, "content": chunk})
                    total_chunks += 1

        # 纯 INSERT 计时
        t_insert = time.time()
        with engine.connect() as conn:
            for r in passage_rows:
                conn.execute(text(
                    "INSERT INTO passages (qa_id,title,content) VALUES (:qa_id,:title,:content)"
                ), r)
            for r in chunk_rows:
                conn.execute(text(
                    "INSERT INTO passage_chunks (qa_id,title,chunk_index,content) "
                    "VALUES (:qa_id,:title,:chunk_index,:content)"
                ), r)
            conn.commit()
        insert_time = time.time() - t_insert
        self.logger.info(f"[SQLAgent-ClapNQ] {len(passage_rows)} passages, {total_chunks} chunks, {total_tokens} tokens, insert_time={insert_time:.2f}s")
        return total_tokens, insert_time

    # ---- FinanceBench ----

    def _ingest_financebench(self, engine, samples: List[StandardDoc]):
        schema = """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_name TEXT UNIQUE, company TEXT, gics_sector TEXT,
            doc_type TEXT, doc_period INTEGER, num_pages INTEGER, content TEXT
        );
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_name TEXT, chunk_index INTEGER, page_num INTEGER, content TEXT
        );"""
        self._exec_schema(engine, schema)

        # FinanceBench chunk 参数更大
        cs = max(self.chunk_size, 1000)
        co = max(self.chunk_overlap, 150)

        # 加载文档元数据
        doc_info = {}
        doc_info_path = os.path.join(os.path.dirname(self.raw_data_path), 'financebench_document_information.jsonl')
        if os.path.exists(doc_info_path):
            with open(doc_info_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        doc_info[item.get('doc_name', '')] = item

        total_chunks = 0
        total_tokens = 0

        # 预处理：读取文档内容并统计 tokens
        doc_rows = []
        chunk_rows = []
        for sample in samples:
            doc_name = sample.sample_id
            content = ''
            for dp in sample.doc_paths:
                content += self._read_document(dp)
            total_tokens += self.count_tokens(content)
            m = doc_info.get(doc_name, {})
            doc_rows.append({"doc_name": doc_name, "company": m.get('company', ''), "gics_sector": m.get('gics_sector', ''),
                             "doc_type": m.get('doc_type', ''), "doc_period": int(m.get('doc_period', 0) or 0),
                             "num_pages": 0, "content": content})
            for ci, chunk in enumerate(self._do_chunk(content, chunk_size=cs, chunk_overlap=co)):
                chunk_rows.append({"doc_name": doc_name, "chunk_index": ci, "page_num": 0, "content": chunk})
                total_chunks += 1
            self.logger.info(f"[SQLAgent-FinanceBench] Prepared doc '{doc_name}' with {total_tokens} tokens and {total_chunks} chunks")

        # 纯 INSERT 计时
        t_insert = time.time()
        with engine.connect() as conn:
            for r in doc_rows:
                conn.execute(text(
                    "INSERT OR IGNORE INTO documents "
                    "(doc_name,company,gics_sector,doc_type,doc_period,num_pages,content) "
                    "VALUES (:doc_name,:company,:gics_sector,:doc_type,:doc_period,:num_pages,:content)"
                ), r)
            for r in chunk_rows:
                conn.execute(text(
                    "INSERT INTO document_chunks (doc_name,chunk_index,page_num,content) "
                    "VALUES (:doc_name,:chunk_index,:page_num,:content)"
                ), r)
            conn.commit()
        insert_time = time.time() - t_insert
        self.logger.info(f"[SQLAgent-FinanceBench] {len(samples)} docs, {total_chunks} chunks, {total_tokens} tokens, insert_time={insert_time:.2f}s")
        return total_tokens, insert_time

    # ---- Generic fallback ----

    def _ingest_generic(self, engine, samples: List[StandardDoc]):
        schema = """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY, content TEXT
        );
        CREATE TABLE IF NOT EXISTS document_chunks (
            doc_id TEXT, chunk_index INTEGER, content TEXT,
            PRIMARY KEY (doc_id, chunk_index)
        );"""
        self._exec_schema(engine, schema)

        total_chunks = 0
        total_tokens = 0

        # 预处理：读取文档内容并统计 tokens
        doc_rows = []
        chunk_rows = []
        for sample in samples:
            for dp in sample.doc_paths:
                content = self._read_document(dp)
                if not content:
                    continue
                total_tokens += self.count_tokens(content)
                doc_id = os.path.splitext(os.path.basename(dp))[0]
                doc_rows.append({"doc_id": doc_id, "content": content})
                for ci, chunk in enumerate(self._do_chunk(content)):
                    chunk_rows.append({"doc_id": doc_id, "chunk_index": ci, "content": chunk})
                    total_chunks += 1

        # 纯 INSERT 计时
        t_insert = time.time()
        with engine.connect() as conn:
            for r in doc_rows:
                conn.execute(text(
                    "INSERT OR REPLACE INTO documents (doc_id,content) VALUES (:doc_id,:content)"
                ), r)
            for r in chunk_rows:
                conn.execute(text(
                    "INSERT OR REPLACE INTO document_chunks (doc_id,chunk_index,content) "
                    "VALUES (:doc_id,:chunk_index,:content)"
                ), r)
            conn.commit()
        insert_time = time.time() - t_insert
        self.logger.info(f"[SQLAgent-Generic] {len(samples)} samples, {total_chunks} chunks, {total_tokens} tokens, insert_time={insert_time:.2f}s")
        return total_tokens, insert_time

    # ---- 辅助方法 ----

    def _exec_schema(self, engine, schema: str):
        with engine.connect() as conn:
            for stmt in schema.strip().split(';'):
                if stmt.strip():
                    conn.execute(text(stmt))
            conn.commit()

    def _load_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_qasper_papers(self) -> List[Dict]:
        """加载 Qasper 数据，JSONL 格式（每行一个 JSON 对象），与原版一致"""
        papers = []
        if os.path.isfile(self.raw_data_path):
            # 单文件：尝试 JSONL
            papers.extend(self._load_jsonl_or_json(self.raw_data_path))
        else:
            for name in ['test.json', 'train.json', 'validation.json',
                          'qasper-dev-v0.3.json', 'qasper-test-v0.3.json']:
                fp = os.path.join(self.raw_data_path, name)
                if os.path.exists(fp):
                    papers.extend(self._load_jsonl_or_json(fp))
        return papers

    def _load_jsonl_or_json(self, filepath: str) -> List[Dict]:
        """加载 JSONL 或 JSON 文件，自动检测格式"""
        items = []
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            f.seek(0)
            if first_line.startswith('{') and not first_line.endswith('}'):
                # 可能是完整 JSON 文件
                data = json.load(f)
                if isinstance(data, dict):
                    for pid, paper in data.items():
                        paper['id'] = pid
                        items.append(paper)
                elif isinstance(data, list):
                    items = data
            else:
                # JSONL 格式
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        return items

    def _load_clapnq_data(self) -> List[Dict]:
        """加载 ClapNQ JSONL 数据"""
        qa_data = []
        base = self.raw_data_path
        patterns = []
        for split in ['dev']:
            for kind in ['answerable']:
                patterns.append(os.path.join(base, 'annotated_data', split, f'clapnq_{split}_{kind}.jsonl'))
        # 也支持直接 JSONL 文件
        if os.path.isfile(base) and base.endswith('.jsonl'):
            patterns = [base]

        for fp in patterns:
            if not os.path.exists(fp):
                continue
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        item.setdefault('qa_id', item.get('id', ''))
                        qa_data.append(item)
        return qa_data

    # ---- retrieve / process / clear / close ----

    def _build_retrieve_prompt(self, question: str, sample_id: str = '',
                                qa_metadata: Optional[Dict] = None) -> str:
        """根据 dataset_name 构建数据集专用的检索 prompt，与原版 LangChain-SQL-Agent 一致"""
        meta = qa_metadata or {}
        ds = self.dataset_name

        retrieve_suffix = (
            "Do NOT answer the question. Only return the relevant raw texts "
            "from the database that could help answer it. "
            "Return each relevant text on a separate line, prefixed with '- '."
        )

        if ds == 'locomo':
            locomo_suffix = (
                "Do NOT answer the question. Only return the relevant raw texts "
                "from the database that could help answer it. "
                "For each result, include the speaker, session_date, and text. "
                "Format: '- [speaker, session_date] text'"
            )
            return (
                f"Find conversations that are relevant to the following question.\n\n"
                f"Question: {question}\n\n"
                f"Note: If the question contains relative time expressions "
                f"(e.g. 'yesterday', 'last week'), query session_date first "
                f"to resolve them to absolute dates before searching.\n\n"
                f"{locomo_suffix}"
            )

        elif ds == 'qasper':
            return (
                f"Find relevant text from the research paper. "
                f"Search in the 'papers', 'sections', and 'section_chunks' tables.\n\n"
                f"Question: {question}\n\n{retrieve_suffix}"
            )

        elif ds == 'hotpotqa':
            return (
                f"Find relevant text from the Wikipedia articles in the database. "
                f"Search in both 'articles' and 'article_chunks' tables.\n\n"
                f"Question: {question}\n\n{retrieve_suffix}"
            )

        elif ds == 'syllabusqa':
            return (
                f"Find relevant text from the syllabus data. "
                f"Search in both 'syllabi' and 'syllabus_chunks' tables.\n\n"
                f"Question: {question}\n\n{retrieve_suffix}"
            )

        elif ds == 'clapnq':
            return (
                f"Find relevant text from the passage data. "
                f"Search in both 'passages' and 'passage_chunks' tables.\n\n"
                f"Question: {question}\n\n{retrieve_suffix}"
            )

        elif ds == 'financebench':
            return (
                f"Find relevant text from the financial document data. "
                f"Search in both 'documents' and 'document_chunks' tables.\n\n"
                f"Question: {question}\n\n{retrieve_suffix}"
            )

        # fallback: 通用 prompt
        return f"Find relevant texts from the database for the following question.\n\nQuestion: {question}\n\n{retrieve_suffix}"

    def retrieve(self, query: str, topk: int = 10, target_uri: str = None,
                 sample_id: str = '', qa_metadata: Optional[Dict] = None) -> SQLAgentResult:
        executor = self._ensure_agent()

        # 设置当前 sample_id，让 tracker 按 sample 分桶记录
        ctx_token = None
        if self._token_tracker and sample_id:
            ctx_token = self._token_tracker.set_sample_id(sample_id)

        # 记录调用前的 token 用量，用于计算本次增量
        usage_before = {"total_prompt_tokens": 0, "total_completion_tokens": 0}
        if self._token_tracker and sample_id:
            usage_before = self._token_tracker.get_usage(sample_id)

        # 构建数据集专用的检索 prompt
        prompt = self._build_retrieve_prompt(query, sample_id, qa_metadata)

        sql_queries = []
        try:
            result = executor.invoke({"input": prompt})
            answer = result.get("output", "")
            # 从中间步骤中提取并记录 SQL 语句
            for step in result.get("intermediate_steps", []):
                action, observation = step
                tool_name = getattr(action, 'tool', '')
                tool_input = getattr(action, 'tool_input', '')
                if tool_name in ('sql_db_query', 'sql_db_write'):
                    sql_queries.append(tool_input)
                    self.logger.info(f"[SQLAgent] Tool: {tool_name} | SQL: {tool_input}")
        except Exception as e:
            self.logger.error(f"SQL Agent retrieve failed: {e}")
            answer = f"[ERROR] {e}"
        finally:
            if ctx_token is not None:
                self._token_tracker.restore_sample_id(ctx_token)

        input_tokens, output_tokens = 0, 0
        if self._token_tracker and sample_id:
            usage_after = self._token_tracker.get_usage(sample_id)
            input_tokens = usage_after.get("total_prompt_tokens", 0) - usage_before.get("total_prompt_tokens", 0)
            output_tokens = usage_after.get("total_completion_tokens", 0) - usage_before.get("total_completion_tokens", 0)

        # 解析 agent 输出：按行拆分为检索到的文本片段
        resources = []
        for line in answer.strip().split('\n'):
            line = line.strip().lstrip('- ')
            if line:
                resources.append(SQLAgentResource(uri=f"source_{len(resources)}", content=line, score=1.0))
        if not resources:
            resources = [SQLAgentResource(uri="sql_agent_empty", content="", score=0.0)]

        return SQLAgentResult(
            resources=resources,
            retrieve_input_tokens=input_tokens,
            retrieve_output_tokens=output_tokens,
            agent_answer='',
            sql_queries=sql_queries,
        )

    def process_retrieval_results(self, search_res: SQLAgentResult):
        retrieved_texts, context_blocks, retrieved_uris = [], [], []
        for r in search_res.resources:
            retrieved_uris.append(r.uri)
            retrieved_texts.append(r.content)
            context_blocks.append(r.content[:2000])
        return retrieved_texts, context_blocks, retrieved_uris

    def clear(self) -> None:
        if self._engine:
            try:
                with self._engine.connect() as conn:
                    conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
                    conn.execute(text("PRAGMA journal_mode=DELETE"))
            except Exception:
                pass
            self._engine.dispose()
            self._engine = None
        self._agent_executor = None
        self._token_tracker = None

        import gc
        gc.collect()

        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except PermissionError:
                import time as _t
                _t.sleep(0.5)
                os.remove(self.db_path)
            self.logger.info(f"[SQLAgent] Deleted database: {self.db_path}")
        for suffix in ['-wal', '-shm']:
            p = self.db_path + suffix
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def close(self):
        if self._engine:
            try:
                with self._engine.connect() as conn:
                    conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
            except Exception:
                pass
            self._engine.dispose()
            self._engine = None
        self._agent_executor = None
        self._token_tracker = None
