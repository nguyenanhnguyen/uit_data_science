"""
legalqa_optimized.py -- LegalQA (UIT DSC2026 Task 2)
Pipeline SOTA: Hybrid Retrieval (BM25 + Dense + PhoRanker) + RAG with Qwen2.5-1.5B 4-bit
Target: METEOR > 0.6 | Optimized for RTX 2050 4GB VRAM | Time-gating safe

USAGE:
    pip install numpy sentence-transformers datasets accelerate nltk rouge_score \
                transformers bitsandbytes torch
    python legalqa_optimized.py

FIXES in this version:
    - load_llm() returns (None, None) instead of None to avoid unpack error
    - Graceful fallback when bitsandbytes is missing
    - Improved chunking for non-Dieu documents (administrative docs)
    - build_train_pairs indexes ALL chunks with so_hieu, not just Dieu chunks
    - Sliding window chunking for documents without Dieu structure
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import re
import json
import math
import time
import random
import zipfile
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, Tuple, List, Dict

import numpy as np


# ==============================================================================
# CONFIG
# ==============================================================================
HERE = Path(__file__).resolve().parent
CONTEXTS_DIR = HERE / "selected-contexts"
TRAIN_PATH = HERE / "train.json"
PUBLIC_PATH = HERE / "public-official.json"
OUT_DIR = HERE
CACHE_DIR = HERE / "model_cache"

TIME_BUDGET_SEC = 3 * 3600
FINETUNE_TIME_BUDGET_SEC = 60 * 60
MIN_TRAIN_PAIRS = 10              # Giam xuong vi nhieu van ban khong co Dieu
MAX_TRAIN_EXAMPLES = 3000

# Retrieval Config
DENSE_MODEL_PRIMARY = "minhquan6203/paraphrase-vietnamese-law"
DENSE_MODEL_FALLBACK = "bkai-foundation-models/vietnamese-bi-encoder"
DENSE_MAX_SEQ_LEN = 256
TRAIN_BATCH_SIZE = 8
ENCODE_BATCH_SIZE = 32

RERANKER_MODEL = "itdainb/PhoRanker"
RERANKER_MAX_LENGTH = 512

TOP_K_RETRIEVE = 100
TOP_K_RERANK = 30
DEV_EVAL_SAMPLE_SIZE = 300

# LLM Config
USE_LLM = True
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MAX_NEW_TOKENS = 384
LLM_BATCH_SIZE = 1
LLM_4BIT = True
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.95

FORCE_EXTRACTIVE = False

_START_TIME = time.time()

def elapsed() -> float:
    return time.time() - _START_TIME

def remaining() -> float:
    return TIME_BUDGET_SEC - elapsed()

def checkpoint(label: str) -> None:
    print(f"[{elapsed()/60:5.1f} phut] {label}  (con lai ~{remaining()/60:.1f} phut)")


# ==============================================================================
# STEP 1 -- Chunk corpus (IMPROVED for mixed document types)
# ==============================================================================
# Regex cho nhieu loai van ban: Dieu, Khoan, Chuong, Muc, hoac chia doan
DIEU_RE = re.compile(r"^[ \t]*Dieu\s+(\d+)[a-zdA-ZD]?[\.\s]", re.MULTILINE)
KHOAN_RE = re.compile(r"^[ \t]*Khoan\s+(\d+)[\.\s]", re.MULTILINE)
CHUONG_RE = re.compile(r"^[ \t]*Chuong\s+[IVX\d]+", re.MULTILINE)
MUC_RE = re.compile(r"^[ \t]*Muc\s+\d+", re.MULTILINE)

SO_HEADER_RE = re.compile(r"So\s*[:：]\s*([0-9A-Za-zDd/\-]+)")
SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zDd]{2,10}(?:-[A-Za-zDd]{2,10})?")
LOAI_VB_CANON = ["Thong tu lien tich", "Nghi dinh", "Luat", "Thong tu", "Quyet dinh",
                 "Phap lenh", "Nghi quyet", "Bo luat", "Chi thi"]
LOAI_PATTERN = re.compile("(" + "|".join(re.escape(x) for x in LOAI_VB_CANON) + ")", re.IGNORECASE)


def extract_vb_info(passage: str):
    m = SO_HEADER_RE.search(passage[:1500])
    so_hieu = m.group(1).strip("., ") if m else ""
    if not (so_hieu and SO_HIEU_RE.fullmatch(so_hieu)):
        m2 = SO_HIEU_RE.search(passage[:1500])
        so_hieu = m2.group(0) if m2 else ""
    m3 = LOAI_PATTERN.search(passage[:200]) or LOAI_PATTERN.search(passage[:800])
    loai_vb = ""
    if m3:
        low = m3.group(1).lower()
        for canon in LOAI_VB_CANON:
            if canon.lower() == low:
                loai_vb = canon
                break
    return loai_vb, so_hieu


def chunk_passage(passage: str, doc_id) -> list:
    """Chunk van ban theo cau truc phap ly. Neu khong co Dieu/Khoan, chia sliding window."""
    # Thu tim Dieu truoc
    matches = list(DIEU_RE.finditer(passage))
    if matches:
        chunks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(passage)
            dieu = m.group(1)
            chunks.append({"id": f"{doc_id}_{dieu}_{i}", "dieu_so": dieu, "loai_vb": "", "so_hieu": "",
                            "text": passage[start:end].strip()})
        return chunks

    # Thu tim Khoan
    matches = list(KHOAN_RE.finditer(passage))
    if matches:
        chunks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(passage)
            khoan = m.group(1)
            chunks.append({"id": f"{doc_id}_k{khoan}_{i}", "dieu_so": khoan, "loai_vb": "", "so_hieu": "",
                            "text": passage[start:end].strip()})
        return chunks

    # Thu tim Chuong/Muc
    matches = list(CHUONG_RE.finditer(passage)) or list(MUC_RE.finditer(passage))
    if matches:
        chunks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(passage)
            chunks.append({"id": f"{doc_id}_sec{i}", "dieu_so": str(i+1), "loai_vb": "", "so_hieu": "",
                            "text": passage[start:end].strip()})
        return chunks

    # Fallback: sliding window chunking cho van ban hanh chinh khong co cau truc ro rang
    # Chia thanh cac doan ~1500 ky tu voi overlap 300 ky tu
    text = passage.strip()
    if len(text) <= 2000:
        return [{"id": f"{doc_id}_0", "dieu_so": "0", "loai_vb": "", "so_hieu": "", "text": text}]

    chunks = []
    window_size = 1500
    overlap = 300
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        # Tim dau cau gan nhat de cat
        if end < len(text):
            for j in range(end, max(end-200, start), -1):
                if text[j-1] in '.\n':
                    end = j
                    break
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"id": f"{doc_id}_w{idx}", "dieu_so": str(idx), "loai_vb": "", "so_hieu": "",
                            "text": chunk_text})
        start = end - overlap
        idx += 1
        if start >= len(text) - overlap:
            break
    return chunks if chunks else [{"id": f"{doc_id}_0", "dieu_so": "0", "loai_vb": "", "so_hieu": "", "text": text}]


def load_corpus(contexts_dir: Path) -> list:
    if not contexts_dir.exists():
        raise FileNotFoundError(f"Khong tim thay {contexts_dir}")
    files = sorted(contexts_dir.glob("context_*.json"))
    if not files:
        nested = contexts_dir / "selected-contexts"
        if nested.exists():
            files = sorted(nested.glob("context_*.json"))
    if not files:
        raise FileNotFoundError(f"Khong tim thay context_*.json trong {contexts_dir}")

    all_chunks, n_no_struct = [], 0
    for fp in files:
        try:
            with fp.open(encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        passage = doc.get("passage")
        if not passage:
            continue
        chunks = chunk_passage(passage, doc["id"])
        # Dem so van ban khong co cau truc Dieu/Khoan/Chuong (chi co 1 chunk va dieu_so=0 hoac sliding)
        has_struct = any(c["dieu_so"] != "0" and not c["id"].endswith("_0") for c in chunks)
        if not has_struct:
            n_no_struct += 1
        loai_vb, so_hieu = extract_vb_info(passage)
        for c in chunks:
            c["loai_vb"], c["so_hieu"] = loai_vb, so_hieu
        all_chunks.extend(chunks)

    pct = round(100 * (1 - n_no_struct / len(files)), 2) if files else 0.0
    print(f"  {len(files)} van ban -> {len(all_chunks)} chunk. {pct}% co cau truc phap ly (Dieu/Khoan/Chuong).")
    return all_chunks


# ==============================================================================
# STEP 2 -- BM25 with numpy (vectorized)
# ==============================================================================
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def tokenize_simple(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, tokenized_docs, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(tokenized_docs)
        self.doc_len = np.array([len(d) for d in tokenized_docs], dtype=np.float64)
        self.avgdl = self.doc_len.mean() if self.N else 0.0

        raw_postings = defaultdict(list)
        for i, doc in enumerate(tokenized_docs):
            for term, f in Counter(doc).items():
                raw_postings[term].append((i, f))

        self.inverted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, postings in raw_postings.items():
            idxs = np.fromiter((p[0] for p in postings), dtype=np.int32, count=len(postings))
            freqs = np.fromiter((p[1] for p in postings), dtype=np.float64, count=len(postings))
            self.inverted[term] = (idxs, freqs)

        df = {t: len(idxs) for t, (idxs, _f) in self.inverted.items()}
        idf_raw = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}
        avg_idf = sum(idf_raw.values()) / len(idf_raw) if idf_raw else 0.0
        eps = 0.25 * avg_idf
        self.idf = {t: (v if v > 0 else eps) for t, v in idf_raw.items()}

    def get_scores(self, query_tokens) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float64)
        for term in set(query_tokens):
            posting = self.inverted.get(term)
            if posting is None:
                continue
            idxs, freqs = posting
            idf = self.idf[term]
            denom = freqs + self.k1 * (1 - self.b + self.b * self.doc_len[idxs] / self.avgdl)
            contrib = idf * freqs * (self.k1 + 1) / denom
            scores[idxs] += contrib
        return scores

    def top_k(self, query_tokens, k: int) -> list:
        scores = self.get_scores(query_tokens)
        return list(np.argsort(-scores)[:k])


# ==============================================================================
# STEP 3 -- Labels from citations in train.json (IMPROVED)
# ==============================================================================
DIEU_CITATION_RE = re.compile(r"Dieu\s+(\d+)\s*[a-zdA-ZD]?\b")


def extract_citations(answer: str) -> list:
    out = []
    for m in DIEU_CITATION_RE.finditer(answer):
        window = answer[m.end(): m.end() + 80]
        so_m = SO_HIEU_RE.search(window)
        if so_m and so_m.start() <= 50:
            out.append((m.group(1), so_m.group(0)))
    return out


def norm_so_hieu(s: str) -> str:
    return s.strip().upper()


def build_train_pairs(train_data: dict, all_chunks: list):
    """Index TAT CA chunk co so_hieu, khong chi chunk co dieu_so != '0'.
    Neu khong tim duoc theo (dieu, so_hieu), thu tim theo so_hieu don le."""
    # Index 1: theo (dieu_so, so_hieu)
    so_hieu_index = {}
    # Index 2: theo so_hieu don le (cho van ban khong chia Dieu)
    so_hieu_only_index = {}

    for c in all_chunks:
        if c["so_hieu"]:
            key_full = (c["dieu_so"], norm_so_hieu(c["so_hieu"]))
            so_hieu_index.setdefault(key_full, c["id"])
            key_only = norm_so_hieu(c["so_hieu"])
            so_hieu_only_index.setdefault(key_only, c["id"])

    positive = {}
    for qid, item in train_data.items():
        found = False
        # Thu tim theo (Dieu, so_hieu)
        for dieu, so_hieu in extract_citations(item["answer"]):
            key = (dieu, norm_so_hieu(so_hieu))
            if key in so_hieu_index:
                positive[qid] = so_hieu_index[key]
                found = True
                break
        if found:
            continue
        # Thu tim theo so_hieu don le trong answer
        for _dieu, so_hieu in extract_citations(item["answer"]):
            key_only = norm_so_hieu(so_hieu)
            if key_only in so_hieu_only_index:
                positive[qid] = so_hieu_only_index[key_only]
                found = True
                break
        if found:
            continue
        # Thu tim so_hieu bat ky trong answer (khong can co Dieu)
        all_so = SO_HIEU_RE.findall(item["answer"])
        for so in all_so:
            key_only = norm_so_hieu(so)
            if key_only in so_hieu_only_index:
                positive[qid] = so_hieu_only_index[key_only]
                break

    chunk_by_id = {c["id"]: c for c in all_chunks}
    return positive, chunk_by_id


# ==============================================================================
# STEP 4 -- Fine-tune dense retriever
# ==============================================================================
def _build_training_rows_semi_hard(train_positive, train_data, chunk_by_id, all_chunks, bm25, n_neg: int = 4):
    rows = []
    n = len(train_positive)
    for i, (qid, pos_id) in enumerate(train_positive.items()):
        question = train_data[qid]["question"]
        pos_text = chunk_by_id[pos_id]["text"]
        token_q = tokenize_simple(question)
        ranked = bm25.top_k(token_q, 60)
        candidates = [all_chunks[i2]["id"] for i2 in ranked[5:30] if all_chunks[i2]["id"] != pos_id]
        neg_ids = []
        if len(candidates) >= n_neg:
            neg_ids = random.sample(candidates, n_neg)
        else:
            neg_ids = candidates[:]
            pool = [c["id"] for c in all_chunks if c["id"] != pos_id and c["id"] not in candidates]
            while len(neg_ids) < n_neg and pool:
                neg_ids.append(random.choice(pool))
        row = {"anchor": question, "positive": pos_text}
        for j, nid in enumerate(neg_ids[:n_neg]):
            row[f"negative_{j+1}"] = chunk_by_id[nid]["text"]
        rows.append(row)
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"    rows: {i+1}/{n}  ({elapsed()/60:.1f} phut)")
    return rows


def _cap_cuda_memory(fraction: float = 0.90) -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [VRAM cap] {fraction*100:.0f}% x {total_gb:.1f}GB = ~{fraction*total_gb:.2f}GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _load_dense_model(device: str):
    from sentence_transformers import SentenceTransformer
    for model_name in [DENSE_MODEL_PRIMARY, DENSE_MODEL_FALLBACK]:
        try:
            print(f"  Loading dense: {model_name} ...")
            model = SentenceTransformer(model_name, device=device, cache_folder=str(CACHE_DIR))
            model.max_seq_length = DENSE_MAX_SEQ_LEN
            print(f"  OK -> {model_name}")
            return model
        except Exception as e:
            print(f"  [Error {model_name}]: {e}")
            continue
    raise RuntimeError("Cannot load any dense model.")


def finetune_or_load_dense(train_positive, train_data, chunk_by_id, all_chunks, bm25):
    import torch
    from sentence_transformers import SentenceTransformer

    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    if cuda_ok:
        print(f"  Device: cuda ({torch.cuda.get_device_name(0)})")
        _cap_cuda_memory()
    else:
        print("  [WARNING] No GPU detected.")

    use_finetune = len(train_positive) >= MIN_TRAIN_PAIRS and remaining() > 5 * 60
    if not use_finetune:
        print(f"  {len(train_positive)} pairs -> zero-shot.")
        return _load_dense_model(device)

    model = _load_dense_model(device)

    try:
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        import accelerate
        if tuple(map(int, accelerate.__version__.split(".")[:2])) < (1, 1):
            raise ImportError(f"accelerate {accelerate.__version__} too old")
    except ImportError as e:
        print(f"  [MISSING PKG] {e} -> zero-shot.")
        return model

    if len(train_positive) > MAX_TRAIN_EXAMPLES:
        sampled_qids = random.sample(list(train_positive.keys()), MAX_TRAIN_EXAMPLES)
        train_positive_used = {qid: train_positive[qid] for qid in sampled_qids}
        print(f"  Sampled {MAX_TRAIN_EXAMPLES}/{len(train_positive)} pairs.")
    else:
        train_positive_used = train_positive

    print("  Building training rows (semi-hard negatives) ...")
    rows = _build_training_rows_semi_hard(train_positive_used, train_data, chunk_by_id, all_chunks, bm25)
    dataset = Dataset.from_list(rows)
    print(f"  Rows: {len(dataset)}")

    batch_size = TRAIN_BATCH_SIZE
    for attempt in range(3):
        try:
            loss = MultipleNegativesRankingLoss(model)
            calib_steps = min(10, max(1, len(dataset) // batch_size))
            calib_args = SentenceTransformerTrainingArguments(
                output_dir="dense_finetuned_tmp", max_steps=calib_steps,
                per_device_train_batch_size=batch_size, logging_steps=calib_steps + 1,
                save_strategy="no", report_to=[], disable_tqdm=True,
                fp16=(device == "cuda"),
            )
            calib_start = time.time()
            print("  Calibrating ...")
            SentenceTransformerTrainer(model=model, args=calib_args, train_dataset=dataset, loss=loss).train()
            calib_time = (time.time() - calib_start) / calib_steps

            budget_left = min(remaining() - 5 * 60, FINETUNE_TIME_BUDGET_SEC - (time.time() - calib_start))
            max_steps = max(0, int(budget_left / max(calib_time, 1e-6)))
            max_steps = min(max_steps, (len(dataset) // batch_size) * 8)
            print(f"  Calib: ~{calib_time:.2f}s/step -> max_steps={max_steps} (batch={batch_size})")

            if max_steps > 0:
                args = SentenceTransformerTrainingArguments(
                    output_dir="dense_finetuned", max_steps=max_steps,
                    per_device_train_batch_size=batch_size, learning_rate=2e-5,
                    warmup_steps=0.05, lr_scheduler_type="cosine",
                    logging_steps=max(1, max_steps // 20), save_strategy="no", report_to=[],
                    fp16=(device == "cuda"),
                )
                SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss).train()
            break
        except (RuntimeError, ImportError) as e:
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower() and batch_size > 1:
                print(f"  [OOM] batch={batch_size} -> {batch_size//2}")
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            raise

    model.save_pretrained("dense_finetuned")
    model = model.to(device)
    print(f"  [Device check] {next(model.parameters()).device}")
    return model


# ==============================================================================
# STEP 5 -- Encode corpus + RRF fusion
# ==============================================================================
def encode_corpus(model, all_chunks: list):
    import torch
    if torch.cuda.is_available():
        _cap_cuda_memory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    print(f"  [Encode] device: {next(model.parameters()).device}{' fp16' if device=='cuda' else ''}")

    texts = [c["text"] for c in all_chunks]
    batch_size = ENCODE_BATCH_SIZE
    while True:
        try:
            embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                       show_progress_bar=True, normalize_embeddings=True,
                                       device=device)
            return embeddings
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                print(f"  [OOM] encode batch={batch_size} -> {batch_size//2}")
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            raise


def rrf_retrieve(question: str, bm25: BM25, dense_model, dense_embeddings, all_chunks, top_k: int = TOP_K_RETRIEVE):
    bm25_ranked = bm25.top_k(tokenize_simple(question), top_k)
    q_emb = dense_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    dense_scores = dense_embeddings @ q_emb
    dense_ranked = list(np.argsort(-dense_scores)[:top_k])

    bm25_rank_map = {idx: r for r, idx in enumerate(bm25_ranked)}
    dense_rank_map = {idx: r for r, idx in enumerate(dense_ranked)}
    all_idx = set(bm25_ranked) | set(dense_ranked)
    rrf = {i: 1 / (60 + bm25_rank_map.get(i, top_k + 1)) + 1 / (60 + dense_rank_map.get(i, top_k + 1))
           for i in all_idx}
    ranked = sorted(rrf, key=rrf.get, reverse=True)
    return [all_chunks[i] for i in ranked]


# ==============================================================================
# STEP 5b -- PhoRanker Cross-Encoder Reranker
# ==============================================================================
def load_reranker():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    try:
        print(f"  Loading PhoRanker {RERANKER_MODEL} ...")
        tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, cache_dir=str(CACHE_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL, cache_dir=str(CACHE_DIR))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        if device == "cuda":
            model = model.half()
        model.eval()
        print(f"  PhoRanker ready on {next(model.parameters()).device}")
        return model, tokenizer
    except Exception as e:
        print(f"  [WARNING] Cannot load PhoRanker ({e}) -> skip rerank.")
        return None, None


def rerank(question: str, candidates: list, reranker_model, reranker_tokenizer,
           max_candidates: int = TOP_K_RERANK, max_length: int = RERANKER_MAX_LENGTH) -> list:
    import torch
    if reranker_model is None or not candidates:
        return candidates
    subset = candidates[:max_candidates]
    device = next(reranker_model.parameters()).device
    pairs = [[question, c["text"]] for c in subset]
    try:
        with torch.no_grad():
            inputs = reranker_tokenizer(pairs, padding=True, truncation=True,
                                         return_tensors="pt", max_length=max_length).to(device)
            scores = reranker_model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()
        order = np.argsort(-scores)
        reranked = [subset[i] for i in order]
        return reranked + candidates[max_candidates:]
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("  [OOM rerank] skip.")
            torch.cuda.empty_cache()
            return candidates
        raise


# ==============================================================================
# STEP 5c -- LLM Loader (FIXED: return (None, None) instead of None)
# ==============================================================================
def load_llm() -> Tuple[Optional, Optional]:
    if not USE_LLM or FORCE_EXTRACTIVE:
        return None, None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        bnb_config = None
        if LLM_4BIT and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print(f"  Loading LLM {LLM_MODEL} (4bit=True) ...")
            except ImportError:
                print("  [WARNING] bitsandbytes not installed. Run: pip install -U bitsandbytes>=0.46.1")
                print("  -> Loading LLM in fp16 instead (may OOM on 4GB).")
                bnb_config = None
        else:
            print(f"  Loading LLM {LLM_MODEL} (4bit=False) ...")

        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True, cache_dir=str(CACHE_DIR))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),
        )
        model.eval()
        device = next(model.parameters()).device
        print(f"  LLM ready on {device} (4bit={bnb_config is not None})")
        return model, tokenizer
    except Exception as e:
        print(f"  [WARNING] Cannot load LLM ({e}) -> use extractive.")
        return None, None


def build_prompt(question: str, chunks: list) -> str:
    context_parts = []
    for c in chunks:
        header = f"Dieu {c['dieu_so']}"
        if c["loai_vb"]:
            header += f" {c['loai_vb']}"
        if c["so_hieu"]:
            header += f" so {c['so_hieu']}"
        context_parts.append(f"{header}:\n{c['text']}")
    context = "\n\n".join(context_parts)

    prompt = (
        "Ban la chuyen gia phap luat Viet Nam. Dua tren cac dieu luat duoc cung cap ben duoi, "
        "hay tra loi cau hoi mot cach chinh xac, day du va suc tich. "
        "Neu co nhieu dieu luat lien quan, hay trinh bay theo thu tu logic. "
        "Luon neu ro can cu phap ly (Dieu, Nghi dinh, Thong tu...) trong cau tra loi.\n\n"
        f"Cac dieu luat tham khao:\n{context}\n\n"
        f"Cau hoi: {question}\n\n"
        "Tra loi:"
    )
    return prompt


def generate_answer_llm(question: str, chunks: list, llm_model, llm_tokenizer) -> str:
    import torch
    prompt = build_prompt(question, chunks)
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            do_sample=(LLM_TEMPERATURE > 0),
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = llm_tokenizer.decode(generated, skip_special_tokens=True).strip()
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer


# ==============================================================================
# STEP 6 -- Answer generation
# ==============================================================================
_DIEU_PREFIX_STRIP_RE = re.compile(r"^\s*Dieu\s+\d+[a-zdA-ZD]?\.?\s*", re.IGNORECASE)


def render_extractive(selected_chunks: list, top_n: int) -> str:
    parts, seen = [], set()
    for c in selected_chunks:
        if c["id"] in seen or len(parts) >= top_n:
            continue
        seen.add(c["id"])
        loai_vb = c["loai_vb"] or "van ban"
        so_hieu = c["so_hieu"] or ""
        dieu = c["dieu_so"]
        lead = (f"Can cu Dieu {dieu} {loai_vb} {so_hieu} quy dinh nhu sau:"
                if dieu != "0" else f"Can cu {loai_vb} {so_hieu} quy dinh nhu sau:")
        body = _DIEU_PREFIX_STRIP_RE.sub("", c["text"], count=1) if dieu != "0" else c["text"]
        parts.append(f"{lead}\n{body}")
    return "\n\n".join(parts)


def answer_question(question: str, bm25, dense_model, dense_embeddings, all_chunks, top_n: int,
                     reranker_model=None, reranker_tokenizer=None,
                     llm_model=None, llm_tokenizer=None,
                     use_llm: bool = True) -> str:
    ranked = rrf_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks)
    if not ranked:
        return "Khong tim thay thong tin phap ly cho cau hoi nay."
    if reranker_model is not None:
        ranked = rerank(question, ranked, reranker_model, reranker_tokenizer)

    if use_llm and llm_model is not None:
        try:
            return generate_answer_llm(question, ranked[:top_n], llm_model, llm_tokenizer)
        except Exception as e:
            print(f"    [LLM error] fallback extractive: {e}")
            return render_extractive(ranked, top_n)
    else:
        return render_extractive(ranked, top_n)


# ==============================================================================
# STEP 7 -- Dev-eval + Time calibration
# ==============================================================================
def try_dev_eval(bm25, dense_model, dense_embeddings, all_chunks, train_data,
                  reranker_model=None, reranker_tokenizer=None,
                  llm_model=None, llm_tokenizer=None) -> tuple:
    try:
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        from nltk.translate.meteor_score import meteor_score
        from rouge_score import rouge_scorer
    except Exception as e:
        print(f"  Skip dev-eval (missing lib: {e}). -> top_n=3, no rerank, no LLM.")
        return 3, False, False, 0.0

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    random.seed(42)
    n_sample = min(DEV_EVAL_SAMPLE_SIZE, len(train_data))
    ids = random.sample(list(train_data.keys()), n_sample)

    # Calib LLM speed
    llm_time_per_q = 0.0
    if llm_model is not None and USE_LLM:
        print("  Calibrating LLM speed on 5 samples ...")
        t0 = time.time()
        for qid in ids[:5]:
            _ = answer_question(train_data[qid]["question"], bm25, dense_model, dense_embeddings,
                                all_chunks, 3, reranker_model, reranker_tokenizer,
                                llm_model, llm_tokenizer, use_llm=True)
        llm_time_per_q = (time.time() - t0) / 5
        print(f"    ~{llm_time_per_q:.2f}s/question (LLM)")

    configs = []
    configs.append(("Extractive", False, False))
    if reranker_model is not None:
        configs.append(("Extractive+Rerank", True, False))
    if llm_model is not None and USE_LLM:
        configs.append(("LLM+Rerank", True, True))

    best_cfg = ("Extractive", 3, False, False, -1.0)
    for label, use_rerank, use_llm in configs:
        print(f"  --- {label} ---")
        rr_m = reranker_model if use_rerank else None
        rr_t = reranker_tokenizer if use_rerank else None

        ranked_cache = {}
        for qid in ids:
            ranked = rrf_retrieve(train_data[qid]["question"], bm25, dense_model, dense_embeddings, all_chunks)
            if use_rerank and rr_m is not None and ranked:
                ranked = rerank(train_data[qid]["question"], ranked, rr_m, rr_t)
            ranked_cache[qid] = ranked

        for top_n in (1, 2, 3, 4, 5):
            ms, rs = [], []
            for qid in ids:
                ranked = ranked_cache[qid]
                if use_llm and llm_model is not None:
                    pred = generate_answer_llm(train_data[qid]["question"], ranked[:top_n], llm_model, llm_tokenizer)
                else:
                    pred = render_extractive(ranked, top_n) if ranked else "Khong tim thay thong tin."
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms) / len(ms), sum(rs) / len(rs)
            print(f"    top_n={top_n}  METEOR={m:.4f}  ROUGE-L={r:.4f}")
            if m > best_cfg[4]:
                best_cfg = (label, top_n, use_rerank, use_llm, m)

    label, best_n, best_use_rerank, best_use_llm, best_m = best_cfg
    print(f"  => Pick: {label} | top_n={best_n} | rerank={best_use_rerank} | llm={best_use_llm} | METEOR={best_m:.4f}")

    can_use_llm_for_all = False
    if best_use_llm and llm_time_per_q > 0:
        n_public = 1000
        est_time = n_public * llm_time_per_q
        buffer = 15 * 60
        if est_time + elapsed() + buffer < TIME_BUDGET_SEC:
            can_use_llm_for_all = True
            print(f"  [Time-gate] Est LLM for 1000 Q: ~{est_time/60:.1f} min -> ENOUGH time.")
        else:
            print(f"  [Time-gate] Est LLM: ~{est_time/60:.1f} min -> NOT ENOUGH, fallback extractive.")
    return best_n, best_use_rerank, can_use_llm_for_all, llm_time_per_q


def measure_retrieval_recall(bm25, dense_model, dense_embeddings, all_chunks, train_data,
                              train_positive, reranker_model=None, reranker_tokenizer=None,
                              sample_size: int = 300) -> None:
    if not train_positive:
        print("  [WARNING] No train_positive pairs available -> skip Recall@k measurement.")
        return
    ids = [qid for qid in random.sample(list(train_positive.keys()),
                                          min(sample_size, len(train_positive)))]
    print(f"  Recall@k on {len(ids)} questions ...")
    ks = [1, 3, 5, 10, 30, 100]
    hits_rrf = {k: 0 for k in ks}
    hits_rerank = {k: 0 for k in ks} if reranker_model is not None else None

    for qid in ids:
        question = train_data[qid]["question"]
        pos_id = train_positive[qid]
        ranked = rrf_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks)
        ranked_ids = [c["id"] for c in ranked]
        for k in ks:
            if pos_id in ranked_ids[:k]:
                hits_rrf[k] += 1
        if reranker_model is not None and ranked:
            reranked = rerank(question, ranked, reranker_model, reranker_tokenizer)
            reranked_ids = [c["id"] for c in reranked]
            for k in ks:
                if pos_id in reranked_ids[:k]:
                    hits_rerank[k] += 1

    n = len(ids)
    print("  --- RRF (no rerank) ---")
    for k in ks:
        print(f"    Recall@{k:<3d} = {hits_rrf[k]}/{n} = {100*hits_rrf[k]/n:.1f}%")
    if hits_rerank is not None:
        print("  --- After PhoRanker ---")
        for k in ks:
            print(f"    Recall@{k:<3d} = {hits_rerank[k]}/{n} = {100*hits_rerank[k]/n:.1f}%")


# ==============================================================================
# STEP 8 -- Package submission
# ==============================================================================
def build_submission(answers: dict, expected_ids: set, out_zip: Path) -> None:
    errors = []
    got = set(answers.keys())
    if got != expected_ids:
        errors.append(f"Key mismatch: missing {len(expected_ids-got)}, extra {len(got-expected_ids)}")
    for qid, ans in answers.items():
        if not isinstance(ans, str) or not ans.strip():
            errors.append(f"[{qid}] empty answer")
    if errors:
        raise ValueError("Invalid submission:\n  - " + "\n  - ".join(errors[:20]))

    normalized = {qid: {"answer": str(ans)} for qid, ans in answers.items()}
    json_path = out_zip.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="submission.json")
    with zipfile.ZipFile(out_zip) as zf:
        assert zf.namelist() == ["submission.json"]
        reloaded = json.loads(zf.read("submission.json").decode("utf-8"))
        assert reloaded == normalized
    print(f"  OK -- {out_zip} ({len(normalized)} answers)")


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    checkpoint("Start")

    print("\n=== Step 1: Chunk corpus ===")
    all_chunks = load_corpus(CONTEXTS_DIR)
    checkpoint("Done chunking")

    print("\n=== Step 2: BM25 index ===")
    tokenized = [tokenize_simple(f"{c.get('loai_vb','')} {c['text']}") for c in all_chunks]
    bm25 = BM25(tokenized)
    checkpoint("Done BM25")

    print("\n=== Step 3: Labels from train.json ===")
    with TRAIN_PATH.open(encoding="utf-8") as f:
        train_data = json.load(f)
    train_positive, chunk_by_id = build_train_pairs(train_data, all_chunks)
    print(f"  Positive pairs: {len(train_positive)}/{len(train_data)}")
    checkpoint("Done labels")

    print("\n=== Step 4: Fine-tune dense retriever ===")
    dense_model = finetune_or_load_dense(train_positive, train_data, chunk_by_id, all_chunks, bm25)
    checkpoint("Done dense retriever")

    print("\n=== Step 5: Encode corpus ===")
    dense_embeddings = encode_corpus(dense_model, all_chunks)
    checkpoint("Done encode")

    print("\n=== Step 5b: Load PhoRanker reranker ===")
    reranker_model, reranker_tokenizer = load_reranker()
    checkpoint("Done PhoRanker")

    print("\n=== Step 5c: Load LLM (Qwen2.5-1.5B 4-bit) ===")
    llm_model, llm_tokenizer = load_llm()
    checkpoint("Done LLM")

    print("\n=== Step 5d: Measure Recall@k ===")
    measure_retrieval_recall(bm25, dense_model, dense_embeddings, all_chunks, train_data,
                              train_positive, reranker_model, reranker_tokenizer)
    checkpoint("Done Recall@k")

    print("\n=== Step 6: Dev-eval choose config + time calibration ===")
    top_n, use_reranker, use_llm_final, llm_sec_per_q = try_dev_eval(
        bm25, dense_model, dense_embeddings, all_chunks, train_data,
        reranker_model, reranker_tokenizer, llm_model, llm_tokenizer
    )
    checkpoint("Done dev-eval")

    print("\n=== Step 7: Generate answers for public-official.json ===")
    with PUBLIC_PATH.open(encoding="utf-8") as f:
        questions = json.load(f)

    rr_m = reranker_model if use_reranker else None
    rr_t = reranker_tokenizer if use_reranker else None
    llm_m = llm_model if use_llm_final else None
    llm_t = llm_tokenizer if use_llm_final else None

    answers = {}
    for i, (qid, item) in enumerate(questions.items()):
        answers[qid] = answer_question(
            item["question"], bm25, dense_model, dense_embeddings, all_chunks, top_n,
            reranker_model=rr_m, reranker_tokenizer=rr_t,
            llm_model=llm_m, llm_tokenizer=llm_t,
            use_llm=use_llm_final
        )
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(questions)}  ({elapsed()/60:.1f} phut)")
    n_empty = sum(1 for a in answers.values() if not a.strip())
    print(f"  Generated {len(answers)} answers, {n_empty} empty")
    checkpoint("Done generation")

    print("\n=== Step 8: Package submission.zip ===")
    build_submission(answers, set(questions.keys()), OUT_DIR / "submission.zip")
    checkpoint(f"FINISHED -- total {elapsed()/60:.1f} min (budget {TIME_BUDGET_SEC/3600:.0f}h)")


if __name__ == "__main__":
    main()