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
        except Exception:
            self.enc = None

        self._engine = None
        self._engine_lock = threading.Lock()
        self._token_tracker = None
        self._agent_executor = None

    # ---- engine / agent 基础设施 ----

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
        from langchain_core.tools import Tool
        from langchain_openai import ChatOpenAI
        try:
            from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
        except ImportError:
            from langchain.agents import AgentExecutor, create_tool_calling_agent
        from src.core.sql_agent_token_tracker import TokenTracker

        engine = self._get_engine()
        db = SQLDatabase(engine=engine)
        self._token_tracker = TokenTracker()
        llm = ChatOpenAI(
            model=self.llm_model, api_key=self._get_api_key(),
            base_url=self.llm_base_url, temperature=0,
            callbacks=[self._token_tracker],
        )
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()

        def _run_write_sql(query: str) -> str:
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    conn.commit()
                    if query.strip().upper().startswith("SELECT"):
                        return str(result.fetchall()[:50])
                    return f"OK – rows affected: {result.rowcount}"
            except Exception as exc:
                return f"Error: {exc}"

        tools.append(Tool(name="sql_db_write",
                          description="Execute INSERT/UPDATE/DELETE SQL. For SELECT use sql_db_query.",
                          func=_run_write_sql))

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a precise SQL database assistant. "
             "Use the provided tools to explore the database schema and run queries. "
             "ALWAYS use the tools to get real data — never fabricate results. "
             "If the answer cannot be determined from the database, say 'Insufficient information'."),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)
        self._agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=self.verbose,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True, return_intermediate_steps=False,
        )
        return self._agent_executor

    def count_tokens(self, t: str) -> int:
        if not t or not self.enc:
            return 0
        return len(self.enc.encode(str(t)))

    def _read_document(self, doc_path: str) -> str:
        ext = os.path.splitext(doc_path)[1].lower()
        if ext == '.pdf':
            from pdfminer.high_level import extract_text
            from markdownify import markdownify
            return markdownify(extract_text(doc_path)).strip()
        with open(doc_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read().strip()

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

        handler(engine, samples)
        return {"time": time.time() - start_time, "input_tokens": 0, "output_tokens": 0}

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

        # 读取原始 JSON
        data = self._load_json(self.raw_data_path)
        sample_ids = {s.sample_id for s in samples}

        with engine.connect() as conn:
            for entry in data:
                sid = str(entry.get('sample_id', ''))
                conv = entry.get('conversation', {})
                for sess_idx in range(1, 100):
                    key = f'session_{sess_idx}'
                    if key not in conv:
                        break
                    session = conv[key]
                    date = session[0].get('date', '') if session else ''
                    for turn_i, turn in enumerate(session):
                        conn.execute(text(
                            "INSERT INTO conversations "
                            "(sample_id,session_id,session_date,turn_number,dia_id,speaker,text) "
                            "VALUES (:a,:b,:c,:d,:e,:f,:g)"
                        ), {"a": sid, "b": sess_idx, "c": date, "d": turn_i,
                            "e": turn.get('dia_id', ''), "f": turn.get('speaker', ''),
                            "g": turn.get('text', '')})

                # session summaries
                for sk, summary in entry.get('session_summary', {}).items():
                    m = re.match(r'session_(\d+)_summary', sk)
                    if m:
                        conn.execute(text(
                            "INSERT INTO session_summaries (sample_id,session_id,summary) "
                            "VALUES (:a,:b,:c)"
                        ), {"a": sid, "b": int(m.group(1)), "c": summary})

                # observations
                for ok, obs_dict in entry.get('observation', {}).items():
                    m = re.match(r'session_(\d+)_observation', ok)
                    if not m:
                        continue
                    sess_id = int(m.group(1))
                    for speaker, obs_list in obs_dict.items():
                        for obs_item in obs_list:
                            observation = obs_item[0] if isinstance(obs_item, list) else str(obs_item)
                            dia_ref = obs_item[1] if isinstance(obs_item, list) and len(obs_item) > 1 else ''
                            if isinstance(dia_ref, list):
                                dia_ref = ', '.join(str(x) for x in dia_ref)
                            conn.execute(text(
                                "INSERT INTO observations "
                                "(sample_id,session_id,speaker,observation,dia_id) "
                                "VALUES (:a,:b,:c,:d,:e)"
                            ), {"a": sid, "b": sess_id, "c": speaker,
                                "d": observation, "e": str(dia_ref)})
            conn.commit()
        self.logger.info(f"[SQLAgent-LoCoMo] Ingested for {len(sample_ids)} samples")

    # ---- HotpotQA ----

    def _ingest_hotpotqa(self, engine, samples: List[StandardDoc]):
        schema = """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wiki_id TEXT, title TEXT, url TEXT, content TEXT
        );
        CREATE TABLE IF NOT EXISTS article_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wiki_id TEXT, title TEXT, chunk_index INTEGER, content TEXT
        );"""
        self._exec_schema(engine, schema)

        data = self._load_json(self.raw_data_path)
        total_chunks = 0
        with engine.connect() as conn:
            for art in data:
                wiki_id = art.get('id', art.get('wiki_id', ''))
                title = art.get('title', '')
                url = art.get('url', '')
                # 处理 text 字段：可能是 2D 数组（段落→句子）
                raw_text = art.get('text', '')
                if isinstance(raw_text, list):
                    paragraphs = []
                    for para in raw_text:
                        if isinstance(para, list):
                            paragraphs.append(' '.join(str(s) for s in para))
                        else:
                            paragraphs.append(str(para))
                    content = '\n\n'.join(paragraphs)
                else:
                    content = str(raw_text)
                content = re.sub(r'<[^>]+>', '', content)

                conn.execute(text(
                    "INSERT INTO articles (wiki_id,title,url,content) VALUES (:a,:b,:c,:d)"
                ), {"a": wiki_id, "b": title, "c": url, "d": content})

                for ci, chunk in enumerate(_chunk_text(content, self.chunk_size, self.chunk_overlap)):
                    conn.execute(text(
                        "INSERT INTO article_chunks (wiki_id,title,chunk_index,content) "
                        "VALUES (:a,:b,:c,:d)"
                    ), {"a": wiki_id, "b": title, "c": ci, "d": chunk})
                    total_chunks += 1
            conn.commit()
        self.logger.info(f"[SQLAgent-HotpotQA] {len(data)} articles, {total_chunks} chunks")

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
        total_chunks = 0

        with engine.connect() as conn:
            for paper in papers:
                pid = paper['id']
                if sample_ids and pid not in sample_ids:
                    continue
                conn.execute(text(
                    "INSERT OR IGNORE INTO papers (paper_id,title,abstract) VALUES (:a,:b,:c)"
                ), {"a": pid, "b": paper.get('title', ''), "c": paper.get('abstract', '')})

                ft = paper.get('full_text', {})
                sec_names = ft.get('section_name', [])
                paragraphs = ft.get('paragraphs', [])
                for i, (sname, paras) in enumerate(zip(sec_names, paragraphs)):
                    content = '\n'.join(paras) if isinstance(paras, list) else str(paras)
                    conn.execute(text(
                        "INSERT INTO sections (paper_id,section_index,section_name,content) "
                        "VALUES (:a,:b,:c,:d)"
                    ), {"a": pid, "b": i, "c": sname, "d": content})
                    for ci, chunk in enumerate(_chunk_text(content, self.chunk_size, self.chunk_overlap)):
                        conn.execute(text(
                            "INSERT INTO section_chunks (paper_id,section_index,chunk_index,content) "
                            "VALUES (:a,:b,:c,:d)"
                        ), {"a": pid, "b": i, "c": ci, "d": chunk})
                        total_chunks += 1
            conn.commit()
        self.logger.info(f"[SQLAgent-Qasper] {len(papers)} papers, {total_chunks} chunks")

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
        with engine.connect() as conn:
            for sample in samples:
                name = sample.sample_id
                # 读取文档内容
                content = ''
                for dp in sample.doc_paths:
                    content += self._read_document(dp)
                m = meta.get(name, {})
                conn.execute(text(
                    "INSERT INTO syllabi "
                    "(syllabus_name,course,major,area,university,num_pages,content) "
                    "VALUES (:a,:b,:c,:d,:e,:f,:g)"
                ), {"a": name, "b": m.get('course', ''), "c": m.get('major', ''),
                    "d": m.get('area', ''), "e": m.get('university', ''),
                    "f": int(m.get('num_pages', 0) or 0), "g": content})

                for ci, chunk in enumerate(_chunk_text(content, self.chunk_size, self.chunk_overlap)):
                    conn.execute(text(
                        "INSERT INTO syllabus_chunks (syllabus_name,chunk_index,content) "
                        "VALUES (:a,:b,:c)"
                    ), {"a": name, "b": ci, "c": chunk})
                    total_chunks += 1
            conn.commit()
        self.logger.info(f"[SQLAgent-SyllabusQA] {len(samples)} syllabi, {total_chunks} chunks")

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

        # ClapNQ 的 doc_paths 指向文档文件，但原始数据在 JSONL 中
        # 从 raw_data_path 读取所有 JSONL
        qa_data = self._load_clapnq_data()
        sample_ids = {s.sample_id for s in samples}
        total_chunks = 0

        with engine.connect() as conn:
            for qa in qa_data:
                qa_id = qa.get('qa_id', qa.get('id', ''))
                if sample_ids and qa_id not in sample_ids:
                    continue
                for pg in qa.get('passages', []):
                    title = pg.get('title', '')
                    content = pg.get('text', '')
                    conn.execute(text(
                        "INSERT INTO passages (qa_id,title,content) VALUES (:a,:b,:c)"
                    ), {"a": qa_id, "b": title, "c": content})
                    for ci, chunk in enumerate(_chunk_text(content, self.chunk_size, self.chunk_overlap)):
                        conn.execute(text(
                            "INSERT INTO passage_chunks (qa_id,title,chunk_index,content) "
                            "VALUES (:a,:b,:c,:d)"
                        ), {"a": qa_id, "b": title, "c": ci, "d": chunk})
                        total_chunks += 1
            conn.commit()
        self.logger.info(f"[SQLAgent-ClapNQ] {len(qa_data)} QAs, {total_chunks} chunks")

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
        with engine.connect() as conn:
            for sample in samples:
                doc_name = sample.sample_id
                content = ''
                for dp in sample.doc_paths:
                    content += self._read_document(dp)
                m = doc_info.get(doc_name, {})
                conn.execute(text(
                    "INSERT OR IGNORE INTO documents "
                    "(doc_name,company,gics_sector,doc_type,doc_period,num_pages,content) "
                    "VALUES (:a,:b,:c,:d,:e,:f,:g)"
                ), {"a": doc_name, "b": m.get('company', ''), "c": m.get('gics_sector', ''),
                    "d": m.get('doc_type', ''), "e": int(m.get('doc_period', 0) or 0),
                    "f": 0, "g": content})

                for ci, chunk in enumerate(_chunk_text(content, cs, co)):
                    conn.execute(text(
                        "INSERT INTO document_chunks (doc_name,chunk_index,page_num,content) "
                        "VALUES (:a,:b,:c,:d)"
                    ), {"a": doc_name, "b": ci, "c": 0, "d": chunk})
                    total_chunks += 1
            conn.commit()
        self.logger.info(f"[SQLAgent-FinanceBench] {len(samples)} docs, {total_chunks} chunks")

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
        with engine.connect() as conn:
            for sample in samples:
                for dp in sample.doc_paths:
                    content = self._read_document(dp)
                    if not content:
                        continue
                    doc_id = os.path.splitext(os.path.basename(dp))[0]
                    conn.execute(text(
                        "INSERT OR REPLACE INTO documents (doc_id,content) VALUES (:a,:b)"
                    ), {"a": doc_id, "b": content})
                    for ci, chunk in enumerate(_chunk_text(content, self.chunk_size, self.chunk_overlap)):
                        conn.execute(text(
                            "INSERT OR REPLACE INTO document_chunks (doc_id,chunk_index,content) "
                            "VALUES (:a,:b,:c)"
                        ), {"a": doc_id, "b": ci, "c": chunk})
                        total_chunks += 1
            conn.commit()
        self.logger.info(f"[SQLAgent-Generic] {len(samples)} samples, {total_chunks} chunks")

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
        """加载 Qasper 数据，支持单文件或目录下多 split"""
        papers = []
        if os.path.isfile(self.raw_data_path):
            data = self._load_json(self.raw_data_path)
            if isinstance(data, dict):
                for pid, paper in data.items():
                    paper['id'] = pid
                    papers.append(paper)
            else:
                papers = data
        else:
            for name in ['test.json', 'train.json', 'validation.json',
                          'qasper-dev-v0.3.json', 'qasper-test-v0.3.json']:
                fp = os.path.join(self.raw_data_path, name)
                if os.path.exists(fp):
                    data = self._load_json(fp)
                    if isinstance(data, dict):
                        for pid, paper in data.items():
                            paper['id'] = pid
                            papers.append(paper)
                    else:
                        papers.extend(data)
        return papers

    def _load_clapnq_data(self) -> List[Dict]:
        """加载 ClapNQ JSONL 数据"""
        qa_data = []
        base = self.raw_data_path
        patterns = []
        for split in ['train', 'dev']:
            for kind in ['answerable', 'unanswerable']:
                patterns.append(os.path.join(base, split, f'clapnq_{split}_{kind}.jsonl'))
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

    def retrieve(self, query: str, topk: int = 10, target_uri: str = None) -> SQLAgentResult:
        executor = self._ensure_agent()
        if self._token_tracker:
            self._token_tracker.reset()
        try:
            result = executor.invoke({"input": query})
            answer = result.get("output", "")
        except Exception as e:
            self.logger.error(f"SQL Agent retrieve failed: {e}")
            answer = f"[ERROR] {e}"

        input_tokens, output_tokens = 0, 0
        if self._token_tracker:
            usage = self._token_tracker.total()
            input_tokens = usage.get("total_prompt_tokens", 0)
            output_tokens = usage.get("total_completion_tokens", 0)

        return SQLAgentResult(
            resources=[SQLAgentResource(uri="sql_agent_answer", content=answer, score=1.0)],
            retrieve_input_tokens=input_tokens,
            retrieve_output_tokens=output_tokens,
            agent_answer=answer,
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
