import os
import sys
import json
import time
import logging
import re
import collections
import string
from typing import List, Dict, Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from judge_util import locomo_grader
import openviking as ov

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
if not os.path.exists(DATA_DIR):
    DATA_DIR = BASE_DIR

RAW_DATA_FILE = os.path.join(DATA_DIR,"locomo_data", "locomo10.json")
EXP_DIR = os.path.join(DATA_DIR, "experiment_data")
MANIFEST_PATH = os.path.join(EXP_DIR, "benchmark_manifest.json")
STORE_PATH = os.path.join(DATA_DIR, "viking_store_locomo")

LOG_FILE = os.path.join(DATA_DIR, "benchmark_non_agentic.log")
REPORT_FILE = os.path.join(DATA_DIR, "benchmark_metrics_report.json")
STATE_FILE = os.path.join(DATA_DIR, ".run_state.json") 

MAX_QUERIES_TO_RUN = None  # 控制最多运行的查询个数，设置为 None 则运行全部查询
MAX_WORKERS = 8

RETRIEVAL_ONLY_FILE = os.path.join(DATA_DIR, "retrieval_only_results.json")

# 新增：拆分出来的生成结果文件和最终评测结果文件
GENERATED_ANSWERS_FILE = os.path.join(DATA_DIR, "generated_answers.json")
QA_EVAL_FILE = os.path.join(DATA_DIR, "qa_eval_detailed_results.json")

LLM_CONFIG = {
    "model": "doubao-seed-1-8-251228",
    "temperature": 0,
    "api_key": "76aab4eb-95e0-453e-b898-7934260ae6a1", 
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
}

CONFIG_PATH = os.path.join(BASE_DIR, "ov.conf")
if os.path.exists(CONFIG_PATH):
    os.environ["OPENVIKING_CONFIG_FILE"] = CONFIG_PATH
# ===========================================

# --- tiktoken 初始化 (cl100k_base) ---
try:
    enc = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    print(f"[错误] tiktoken 初始化失败: {e}")
    enc = None

def count_tokens(text: str) -> int:
    """使用 cl100k_base 精确计算 Token 数量"""
    if not text or not enc:
        return 0
    return len(enc.encode(str(text)))


# --- 日志配置 ---
def setup_logger():
    logger = logging.getLogger("Benchmark")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

log = setup_logger()

def log_and_print(msg: str):
    """同时输出到控制台和日志文件"""
    print(msg)
    log.info(msg)


# --- 状态管理 (断点跳过) ---
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"step1_done": False, "step2_done": False, "step3_done": False, "step4_done": False, "step5_done": False, "metrics": {}}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


# --- 指标计算辅助函数 ---
class MetricsCalculator:
    @staticmethod
    def normalize_answer(s):
        s = str(s).replace(',', "") 
        def remove_articles(text): return re.sub(r'\b(a|an|the|and)\b', ' ', text)
        def white_space_fix(text): return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        return white_space_fix(remove_articles(remove_punc(s.lower())))

    @staticmethod
    def check_recall(retrieved_texts: List[str], evidence_ids: List[str]):
        if not evidence_ids: return 0.0 
        hits = sum(1 for ev_id in evidence_ids if any(ev_id in text for text in retrieved_texts))
        return hits / len(evidence_ids)

    @staticmethod
    def calculate_f1(prediction: str, ground_truth: str) -> float:
        pred_tokens = MetricsCalculator.normalize_answer(prediction).split()
        truth_tokens = MetricsCalculator.normalize_answer(ground_truth).split()
        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: return 0.0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def check_refusal(text: str) -> bool:
        refusals = ["not mentioned", "no information", "cannot be answered", "none"]
        return any(r in text.lower() for r in refusals)


# ================= 核心步骤实现 =================

def step1_prepare_data(state):
    log_and_print("\n==================================================")
    log_and_print("=== Step 1: 准备测试数据 (Data Preparation) ===")
    
    if state.get("step1_done"):
        log_and_print("-> 检测到 Step 1 已完成，跳过...")
        return

    os.makedirs(EXP_DIR, exist_ok=True)
    if not os.path.exists(RAW_DATA_FILE):
        log_and_print(f"[错误] 找不到原始数据文件 {RAW_DATA_FILE}")
        return

    with open(RAW_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        dataset = [data] if isinstance(data, dict) else data

    metadata_list = []

    for item in dataset:
        sample_id = item.get("sample_id", "unknown")
        conv = item.get("conversation", {})

        md_lines = [f"# Chat History: {sample_id}"]

        session_idx = 1
        while f"session_{session_idx}" in conv:
            s_key = f"session_{session_idx}"
            dt_key = f"session_{session_idx}_date_time"
            sum_key = f"session_{session_idx}_summary"

            md_lines.append(f"\n## Session {session_idx}")

            session_dt = conv.get(dt_key)
            if session_dt:
                md_lines.append(f"DATE: {session_dt}")

            session_sum = conv.get(sum_key)
            if session_sum:
                md_lines.append(f"SUMMARY: {session_sum}")

            for turn in conv[s_key]:
                spk = turn.get("speaker", "Unknown")
                txt = turn.get("text", "")

                raw_id = turn.get("dia_id") or turn.get("id")
                suffix = f" [{raw_id}]" if raw_id else ""

                md_lines.append(f"**{spk}**: {txt}{suffix}")

            session_idx += 1

        doc_content = "\n".join(md_lines)
        doc_path = os.path.join(EXP_DIR, f"{sample_id}_doc.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc_content)

        qa_list = []
        for q in item.get("qa", []):
            raw_ans = q.get("answer")
            gold = [raw_ans] if isinstance(raw_ans, str) else (raw_ans or ["Not mentioned"])
            qa_list.append({
                "question": q["question"],
                "gold_answers": gold,
                "category": q.get("category"),
                "evidence": q.get("evidence", [])
            })

        metadata_list.append({
            "sample_id": sample_id,
            "doc_path": doc_path,
            "qa_pairs": qa_list
        })

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    log_and_print(f"-> 成功处理 {len(metadata_list)} 个样本并生成 Manifest.")
    state["step1_done"] = True
    save_state(state)


def step2_ingest_data(state):
    log_and_print("\n==================================================")
    log_and_print("=== Step 2: 数据入库 (Data Ingestion) ===")
    
    if state.get("step2_done"):
        log_and_print("-> 检测到 Step 2 已完成，跳过...")
        return

    if not os.path.exists(STORE_PATH): os.makedirs(STORE_PATH)
    
    client = ov.SyncOpenViking(path=STORE_PATH)
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    start_time = time.time()

    log_and_print("-> 开始插入文档到 OpenViking...")
    for sample in samples:
        abs_doc_path = os.path.join(BASE_DIR, sample['doc_path'])
        if os.path.exists(abs_doc_path):
            client.add_resource(abs_doc_path, wait=False)

    log_and_print("-> 等待后台语义处理与向量化 (L0/L1 摘要生成)...")
    client.wait_processed()

    ingest_time = time.time() - start_time

    log_and_print("-> 正在使用 tiktoken (cl100k_base) 盘点入库 Token 消耗...")
    total_input_tokens = 0
    total_output_tokens = 0
    
    for sample in samples:
        abs_doc_path = os.path.join(BASE_DIR, sample['doc_path'])
        if os.path.exists(abs_doc_path):
            with open(abs_doc_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                total_input_tokens += count_tokens(raw_text) * 2 
            
            file_name = os.path.basename(abs_doc_path)
            uri = f"viking://resources/{file_name}"
            try:
                l0_text = client.abstract(uri)
                l1_text = client.overview(uri)
                total_output_tokens += count_tokens(l0_text) + count_tokens(l1_text)
            except Exception:
                pass

    total_ingest_tokens = total_input_tokens + total_output_tokens
    log_and_print(f"-> 入库完成! 耗时: {ingest_time:.2f}s, Input Tokens: {total_input_tokens}, Output Tokens: {total_output_tokens}")
    
    state["metrics"]["insertion_time_total"] = ingest_time
    state["metrics"]["insertion_input_tokens"] = total_input_tokens
    state["metrics"]["insertion_output_tokens"] = total_output_tokens
    state["metrics"]["insertion_tokens_total"] = total_ingest_tokens
    state["step2_done"] = True
    save_state(state)

def step3_retrieval_only(state,
                         sample_ids: List[str] = None,
                         topk: int = 5,
                         max_queries: int = None,
                         print_chars: int = 100):

    log_and_print("\n==================================================")
    log_and_print("=== Step 3R: Retrieval-Only (Structured JSON) ===")

    client = ov.SyncOpenViking(path=STORE_PATH)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # -------- Sample 过滤 --------
    if sample_ids:
        sample_id_set = set(sample_ids)
        samples = [s for s in samples if s.get("sample_id") in sample_id_set]
        log_and_print(f"-> Filtered samples: {len(samples)}")
    else:
        log_and_print(f"-> Using all samples: {len(samples)}")

    total_queries = 0
    evidence_hit_count = 0
    recall_sum = 0.0

    all_results = []

    for sample in samples:
        sid = sample.get("sample_id", "unknown")

        for qa in sample.get("qa_pairs", []):
            if max_queries and total_queries >= max_queries:
                break

            q_text = qa.get("question", "")
            golds = qa.get("gold_answers", [])
            evidence = qa.get("evidence", []) or []

            # 统一格式
            if not isinstance(golds, list):
                golds = [str(golds)]
            else:
                golds = [str(g) for g in golds]

            # -------- 检索 --------
            t0 = time.time()
            target_uri = f"viking://resources/{sid}_doc"
            results = client.find(query=q_text, limit=topk, target_uri=target_uri)
            latency = time.time() - t0

            retrieved_texts = []
            topk_data = []

            for rank, r in enumerate(results.resources, start=1):
                uri = getattr(r, "uri", "")
                score = getattr(r, "score", 0.0)
                is_leaf = getattr(r, "is_leaf", False)

                if is_leaf:
                    try:
                        content = str(client.read(uri))
                    except Exception:
                        content = getattr(r, "abstract", "") or ""
                else:
                    content = f"{getattr(r, 'abstract', '')}\n{getattr(r, 'overview', '')}"

                retrieved_texts.append(content)

                topk_data.append({
                    "rank": rank,
                    "uri": uri,
                    "score": score,
                    "snippet": content[:print_chars]
                })

            # -------- 指标计算 --------
            recall = MetricsCalculator.check_recall(retrieved_texts, evidence)
            evidence_hit = 1 if recall > 0 else 0

            merged = "\n".join(retrieved_texts).lower()
            gold_in_topk = 1 if any(
                g.lower() in merged
                for g in golds
                if g and g != "Not mentioned"
            ) else 0

            # -------- 累积统计 --------
            total_queries += 1
            recall_sum += recall
            evidence_hit_count += evidence_hit

            # -------- 结构化记录 --------
            all_results.append({
                "sample_id": sid,
                "question": q_text,
                "gold_answers": golds,
                "evidence_ids": evidence,
                "retrieval_latency_sec": latency,
                "topk": topk_data,
                "metrics": {
                    "evidence_recall": recall,
                    "evidence_hit": evidence_hit,
                    "gold_in_topk_weak": gold_in_topk
                }
            })

        if max_queries and total_queries >= max_queries:
            break

    # -------- Summary --------
    avg_recall = recall_sum / total_queries if total_queries else 0
    hit_rate = evidence_hit_count / total_queries if total_queries else 0

    summary = {
        "topk": topk,
        "total_queries": total_queries,
        "evidence_hit_rate": hit_rate,
        "avg_evidence_recall": avg_recall
    }

    final_output = {
        "dataset": "LocoMo",
        "config": {
            "topk": topk,
            "max_queries": max_queries,
            "filtered_sample_ids": sample_ids
        },
        "summary": summary,
        "results": all_results
    }

    # -------- 保存 JSON --------
    with open(RETRIEVAL_ONLY_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    log_and_print(f"\n[Saved] Retrieval-only results saved to:")
    log_and_print(f"{RETRIEVAL_ONLY_FILE}")

    state.setdefault("metrics", {})
    state["metrics"]["retrieval_only"] = summary
    save_state(state)


def step3_generate_answers(state, 
                           sample_ids: List[str] = None, 
                           topk: int = 5, 
                           max_queries: int = None):
    """
    Step 3 拆分：只负责检索并使用大模型生成答案，但不进行结果打分（Accuracy 和 F1）
    将生成的答案信息保存到 GENERATED_ANSWERS_FILE
    """
    import random
    log_and_print("\n==================================================")
    log_and_print(f"=== Step 3: 检索与回答生成 (Generation) [topk={topk}] ===")

    if state.get("step3_done"):
        log_and_print("-> 检测到 Step 3 已完成，跳过...")
        return

    if max_queries is None:
        max_queries = MAX_QUERIES_TO_RUN

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # -------- Sample 过滤 --------
    if sample_ids:
        sample_id_set = set(sample_ids)
        samples = [s for s in samples if s.get("sample_id") in sample_id_set]
        log_and_print(f"-> Filtered samples: {len(samples)}")

    QA_PROMPT = """Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""
    QA_PROMPT_CAT_5 = """Based on the above context, answer the following question.

Question: {} Short answer:
"""
    MISSING_RULE = "If no information is available to answer the question, write 'Not mentioned'."

    def get_cat_5_answer(model_prediction: str, answer_key: dict) -> str:
        mp = (model_prediction or "").strip().lower()
        if len(mp) == 1:
            if "a" in mp: return answer_key["a"]
            else: return answer_key["b"]
        elif len(mp) == 3:
            if "(a)" in mp: return answer_key["a"]
            else: return answer_key["b"]
        else:
            return (model_prediction or "").strip()

    tasks = []
    gidx = 0
    for sample_idx, sample in enumerate(samples):
        sample_id = sample.get("sample_id", "unknown")
        for qa_idx, qa in enumerate(sample.get("qa_pairs", [])):
            if max_queries and gidx >= max_queries:
                break
            tasks.append((gidx, sample_idx, qa_idx, sample_id, qa))
            gidx += 1
        if max_queries and gidx >= max_queries:
            break

    total_qas = len(tasks)
    log_and_print(f"-> total QA tasks: {total_qas}, max_workers={MAX_WORKERS}")

    thread_local = threading.local()

    def get_thread_client_llm():
        if not hasattr(thread_local, "client"):
            thread_local.client = ov.SyncOpenViking(path=STORE_PATH)
        if not hasattr(thread_local, "llm"):
            thread_local.llm = ChatOpenAI(**LLM_CONFIG)
        return thread_local.client, thread_local.llm

    def process_one_gen(task):
        global_index, sample_idx, qa_idx, sample_id, qa = task
        client, llm = get_thread_client_llm()

        q_text = qa.get("question", "")
        evidence = qa.get("evidence", []) or []
        category = qa.get("category", "unknown")

        golds = qa.get("gold_answers", []) or []
        if not isinstance(golds, list):
            golds = [str(golds)]
        else:
            golds = [str(g) for g in golds]

        # ---------- 1) Retrieval ----------
        retrieval_t0 = time.time()
        target_uri = "viking://resources"
        results = client.find(query=q_text, limit=topk, target_uri=target_uri)
        retrieval_latency = time.time() - retrieval_t0
        retrieval_input_tokens = count_tokens(q_text)

        retrieved_texts = []
        context_blocks = []
        topk_data = []

        for rank, r in enumerate(results.resources, start=1):
            uri = getattr(r, 'uri', '')
            score = getattr(r, 'score', 0.0)
            is_leaf = getattr(r, 'is_leaf', False)

            if is_leaf:
                try: content = str(client.read(uri))
                except Exception: content = getattr(r, 'abstract', '') or ""
            else:
                content = f"{getattr(r, 'abstract', '')}\n{getattr(r, 'overview', '')}"

            retrieved_texts.append(content)
            clean_content = re.sub(r' \[.*?\]', '', content)
            clean_for_llm = clean_content[:2000]
            context_blocks.append(clean_for_llm)

            topk_data.append({
                "rank": rank,
                "uri": uri,
                "score": score,
                "snippet": clean_content[:500]
            })

        # 计算纯检索召回率（不涉及LLM打分）
        recall = MetricsCalculator.check_recall(retrieved_texts, evidence)

        # ---------- 2) Prompting ----------
        effective_question = q_text
        used_prompt_template = "QA_PROMPT"
        cat5_options = None
        mapped_answer = None

        if category == 2 or str(category) == "2":
            effective_question = q_text + " Use DATE of CONVERSATION to answer with an approximate date."

        if category == 5 or str(category) == "5":
            used_prompt_template = "QA_PROMPT_CAT_5"
            gold = next((str(g).strip() for g in golds if g and str(g).strip()), None)

            if not gold:
                effective_question = q_text
                used_prompt_template = "QA_PROMPT"
            else:
                if random.random() < 0.5:
                    effective_question = q_text + f" Select the correct answer: (a) Not mentioned in the conversation (b) {gold}."
                    answer_key = {"a": "Not mentioned in the conversation", "b": gold}
                else:
                    effective_question = q_text + f" Select the correct answer: (a) {gold} (b) Not mentioned in the conversation."
                    answer_key = {"a": gold, "b": "Not mentioned in the conversation"}
                cat5_options = {"a": answer_key["a"], "b": answer_key["b"]}

        combined_context = "\n\n".join(context_blocks)
        if used_prompt_template == "QA_PROMPT_CAT_5":
            question_part = QA_PROMPT_CAT_5.format(effective_question)
        else:
            question_part = QA_PROMPT.format(effective_question)

        full_prompt = combined_context + "\n\n" + MISSING_RULE + "\n\n" + question_part
        qa_input_tokens = count_tokens(full_prompt)

        # ---------- 3) LLM 生成 ----------
        ans_raw = ""
        last_err = None
        for attempt in range(3):
            try:
                ans_raw = llm.invoke([HumanMessage(content=full_prompt)]).content
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))

        if last_err is not None:
            ans_raw = f"ERROR: {last_err}"
        ans_raw = (ans_raw or "").strip()

        if (category == 5 or str(category) == "5") and cat5_options:
            ans = get_cat_5_answer(ans_raw, cat5_options)
            mapped_answer = ans
        else:
            ans = ans_raw

        qa_output_tokens = count_tokens(ans)
        total_query_input_tokens = retrieval_input_tokens + qa_input_tokens

        # 构建输出结构 (由于把打分解耦了，此处填充None占位)
        detailed = {
            "sample_id": sample_id,
            "qa_index": {"sample_idx": sample_idx, "qa_idx": qa_idx},
            "question": q_text,
            "category": category,
            "evidence": evidence,
            "gold_answers": golds,
            "retrieval": {
                "target_uri": target_uri,
                "query_used": q_text,
                "topk": topk_data,
                "latency_sec": retrieval_latency
            },
            "llm": {
                "effective_question": effective_question,
                "prompt_template": used_prompt_template,
                "cat5_options": cat5_options,
                "raw_answer": ans_raw,
                "final_answer": ans,
                "mapped_answer": mapped_answer
            },
            "token_usage": {
                "retrieval_input_tokens": retrieval_input_tokens,
                "llm_input_tokens": qa_input_tokens,
                "llm_output_tokens": qa_output_tokens,
                "total_input_tokens": total_query_input_tokens,
                "total_tokens": total_query_input_tokens + qa_output_tokens
            },
            "metrics": {
                "Recall": recall,
                "Latency": retrieval_latency,
                "F1": None,          # 留到 Step 4 计算
                "Accuracy": None     # 留到 Step 4 计算
            }
        }

        return {
            "_global_index": global_index,
            "detailed": detailed
        }

    # 执行并发生成任务
    detailed_results = []
    pbar = tqdm(total=total_qas, desc="Step3 Generate (mt)", unit="q")
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(process_one_gen, t) for t in tasks]
            for fu in as_completed(futures):
                res = fu.result()
                detailed_results.append(res)
                pbar.update(1)
    finally:
        pbar.close()

    # 根据原顺序进行排序并保存
    detailed_results.sort(key=lambda x: x["_global_index"])
    final_output_list = [r["detailed"] for r in detailed_results]

    summary = {
        "dataset": "LocoMo",
        "total_queries": total_qas,
        "config": {
            "topk": topk,
            "filtered_sample_ids": sample_ids
        }
    }

    with open(GENERATED_ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": final_output_list}, f, indent=2, ensure_ascii=False)

    log_and_print(f"\n-> 检索与答案生成完毕，结果已保存至: {GENERATED_ANSWERS_FILE}")

    state["step3_done"] = True
    save_state(state)


def step4_evaluate_answers(state):
    """
    Step 4 拆分：基于 Step 3 保存的 `generated_answers.json`，使用 LLM Grader 等工具对答案进行评分（计算 Accuracy, F1）
    """
    log_and_print("\n==================================================")
    log_and_print("=== Step 4: 结果评测与打分 (Evaluation) ===")

    if state.get("step4_done"):
        log_and_print("-> 检测到 Step 4 已完成，跳过...")
        return

    if not os.path.exists(GENERATED_ANSWERS_FILE):
        log_and_print(f"[错误] 找不到回答生成文件: {GENERATED_ANSWERS_FILE}")
        return

    with open(GENERATED_ANSWERS_FILE, "r", encoding="utf-8") as f:
        gen_data = json.load(f)

    results = gen_data.get("results", [])
    # 过滤掉对抗性问题
    results[:] = [r for r in results if r.get('category') != 5]
    total_qas = len(results)

    thread_local = threading.local()

    def get_thread_llm():
        if not hasattr(thread_local, "llm"):
            thread_local.llm = ChatOpenAI(**LLM_CONFIG)
        return thread_local.llm

    def process_one_eval(idx, item):
        llm = get_thread_llm()
        
        q_text = item["question"]
        golds = item["gold_answers"]
        ans = item["llm"]["final_answer"]

        # ---- F1 指标计算 ----
        f1 = max((MetricsCalculator.calculate_f1(ans, gt) for gt in golds), default=0.0) if golds else 0.0

        # ---- Accuracy 指标计算 (调用 locomo_grader) ----
        gold_answer = "\n".join(f"{i+1}. {g}" for i, g in enumerate(golds))
        if isinstance(llm, ChatOpenAI):
            acc = 1.0 if locomo_grader(llm, LLM_CONFIG.get("model"), q_text, gold_answer, ans) else 0.0
        else:
            acc = 0.0
        if MetricsCalculator.check_refusal(ans) and any(MetricsCalculator.check_refusal(gt) for gt in golds):
            f1, acc = 1.0, 1.0

        # 写回数据
        item["metrics"]["F1"] = f1
        item["metrics"]["Accuracy"] = acc

        # 整理汇总用的记录
        return {
            "_global_index": idx,
            "f1": f1,
            "accuracy": acc,
            "recall": item["metrics"]["Recall"],
            "retrieval_time": item["retrieval"]["latency_sec"],
            "query_input_tokens": item["token_usage"]["total_input_tokens"],
            "query_output_tokens": item["token_usage"]["llm_output_tokens"]
        }

    eval_records = []
    pbar = tqdm(total=total_qas, desc="Step4 Evaluate (mt)", unit="q")
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            # 传递(索引, 字典对象引用)，字典会直接在并发中被修改
            futures = [ex.submit(process_one_eval, i, res) for i, res in enumerate(results)]
            for fu in as_completed(futures):
                eval_records.append(fu.result())
                pbar.update(1)
    finally:
        pbar.close()

    # 聚合汇总
    eval_records.sort(key=lambda x: x["_global_index"])
    
    summary = {
        "dataset": "LocoMo",
        "total_queries": total_qas,
        "avg_f1": sum(r["f1"] for r in eval_records) / total_qas if total_qas else 0.0,
        "avg_recall": sum(r["recall"] for r in eval_records) / total_qas if total_qas else 0.0,
        "avg_accuracy": sum(r["accuracy"] for r in eval_records) / total_qas if total_qas else 0.0,
        "avg_retrieval_time": sum(r["retrieval_time"] for r in eval_records) / total_qas if total_qas else 0.0,
        "avg_query_input_tokens": sum(r["query_input_tokens"] for r in eval_records) / total_qas if total_qas else 0.0,
        "avg_query_output_tokens": sum(r["query_output_tokens"] for r in eval_records) / total_qas if total_qas else 0.0,
        "max_workers": MAX_WORKERS
    }

    with open(QA_EVAL_FILE, "w", encoding="utf-8") as f:
        # results 已经在 process_one_eval 中直接更新上了 F1 和 Accuracy
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    log_and_print(f"\n-> 最终评测结果已保存至: {QA_EVAL_FILE}")
    log_and_print(json.dumps(summary, indent=2, ensure_ascii=False))

    state.setdefault("metrics", {})
    state["metrics"]["retrieval"] = summary
    state["step4_done"] = True
    save_state(state)


def step5_delete_data(state):
    log_and_print("\n==================================================")
    log_and_print("=== Step 5: 数据清理 (Data Deletion) ===")
    
    if state.get("step5_done"):
        log_and_print("-> 检测到 Step 5 已完成，跳过...")
        return

    client = ov.SyncOpenViking(path=STORE_PATH)
    
    start_time = time.time()
    log_and_print("-> 正在执行递归删除...")
    
    client.rm("viking://resources", recursive=True)
    delete_time = time.time() - start_time
    
    delete_input_tokens = 0
    delete_output_tokens = 0
    
    log_and_print(f"-> 删除完成! 耗时: {delete_time:.2f}s, Token成本: 0")
    
    state["metrics"]["deletion_time_total"] = delete_time
    state["metrics"]["deletion_input_tokens"] = delete_input_tokens
    state["metrics"]["deletion_output_tokens"] = delete_output_tokens
    state["step5_done"] = True
    save_state(state)


def generate_final_report(state):
    log_and_print("\n\n############################################################")
    log_and_print("            FINAL METRICS REPORT (NON-AGENTIC)           ")
    log_and_print("############################################################")
    
    m = state.get("metrics", {})
    r = m.get("retrieval", {})
    
    report_data = {
        "Dataset": "LocoMo",
        "Total Queries Evaluated": r.get("total_queries", 0),
        "Performance Metrics": {
            "Average F1 Score": r.get("avg_f1", 0),
            "Average Recall": r.get("avg_recall", 0),
            "Average Accuracy (Hit Rate)": r.get("avg_accuracy", 0)
        },
        "Query Efficiency (Average Per Query)": {
            "Average Retrieval Time (s)": r.get("avg_retrieval_time", 0),
            "Average Input Tokens": r.get("avg_query_input_tokens", 0),
            "Average Output Tokens": r.get("avg_query_output_tokens", 0)
        },
        "Insertion Efficiency (Total Dataset)": {
            "Total Insertion Time (s)": m.get("insertion_time_total", 0),
            "Total Input Tokens": m.get("insertion_input_tokens", 0),
            "Total Output Tokens": m.get("insertion_output_tokens", 0)
        },
        "Deletion Efficiency (Total Dataset)": {
            "Total Deletion Time (s)": m.get("deletion_time_total", 0),
            "Total Input Tokens": m.get("deletion_input_tokens", 0),
            "Total Output Tokens": m.get("deletion_output_tokens", 0)
        }
    }
    
    report_json = json.dumps(report_data, indent=4, ensure_ascii=False)
    log_and_print(report_json)
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
        
    log_and_print(f"\n[完成] 完整报告已保存至: {REPORT_FILE}")


if __name__ == "__main__":
    log_and_print("启动 OpenViking Non-Agentic Benchmark...")
    run_state = load_state()
    
    try:
        step1_prepare_data(run_state)
        step2_ingest_data(run_state)
        
        # 增加控制参数
        step3_generate_answers(
            run_state, 
            sample_ids=["conv-26"],
            topk=5
        )

        step4_evaluate_answers(run_state)
        
        # 若需要清理数据可以开启 Step 5
        # step5_delete_data(run_state)
        
    except KeyboardInterrupt:
        log_and_print("\n[!] 实验被中断。状态已保存，下次运行可从断点继续。")
    except Exception as e:
        log_and_print(f"\n[!] 运行中发生错误: {str(e)}")
        log.exception("Runtime Exception")
    finally:
        if run_state.get("metrics", {}).get("retrieval"):
            generate_final_report(run_state)