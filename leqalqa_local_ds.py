"""
legalqa_advanced.py — LegalQA (UIT DSC2026 Task 2)
- Dense retriever: bkai-foundation-models/vietnamese-bi-encoder (fine-tune)
- Reranker: xlm-roberta-base (cross-encoder, fine-tune)
- Answer generation: template extractive (ghép nguyên văn)
- Tối ưu cho RTX 2050 4GB VRAM, thời gian ~2 giờ.
- Mục tiêu METEOR ≥ 0.50.
"""

from __future__ import annotations
import os
import sys
import re
import json
import math
import time
import random
import zipfile
import pickle
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset

# =========================== CONFIG ===========================
HERE = Path(__file__).resolve().parent
CONTEXTS_DIR = HERE / "selected-contexts"          # thư mục chứa context_*.json
TRAIN_PATH = HERE / "train.json"
PUBLIC_PATH = HERE / "public-official.json"
OUT_ZIP = HERE / "submission.zip"

# ---- Mô hình ----
DENSE_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"   # 135M, fine-tune
RERANKER_MODEL_NAME = "xlm-roberta-base"                           # 278M, fine-tune

# ---- Hyperparameters ----
DENSE_MAX_SEQ_LEN = 256
RERANKER_MAX_SEQ_LEN = 512
ENCODE_BATCH_SIZE = 32
TRAIN_BATCH_SIZE_DENSE = 8          # tự giảm nếu OOM
TRAIN_BATCH_SIZE_RERANKER = 8       # tự giảm nếu OOM
TOP_K_RETRIEVE = 100
TOP_K_RERANK = 10
TOP_N_CANDIDATES = (1, 3, 5, 7)     # các giá trị top_n thử trong dev-eval

# ---- Fine-tune options ----
USE_FINETUNE_DENSE = True
USE_FINETUNE_RERANKER = True
MIN_TRAIN_PAIRS = 50
MAX_TRAIN_EXAMPLES_DENSE = 3000
MAX_TRAIN_EXAMPLES_RERANKER = 1000

# ---- Cache ----
EMBED_CACHE_FILE = HERE / "corpus_embeddings.pkl"

# =========================== UTILITIES ===========================
_START_TIME = time.time()
def elapsed():
    return time.time() - _START_TIME

def checkpoint(label):
    print(f"[{elapsed()/60:5.1f} phút] {label}")

def _cap_cuda_memory(fraction=0.92):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  [VRAM limit] {fraction*100:.0f}% = {fraction*total:.2f}GB")

# =========================== BƯỚC 1: CHUNK CORPUS ===========================
DIEU_RE = re.compile(r"^[ \t]*Điều\s+(\d+)[a-zđA-ZĐ]?[\.\s]", re.MULTILINE)
SO_HEADER_RE = re.compile(r"Số\s*[:：]\s*([0-9A-Za-zĐđ/\-]+)")
SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zĐđ]{2,10}(?:-[A-Za-zĐđ]{2,10})?")
LOAI_VB_CANON = ["Thông tư liên tịch", "Nghị định", "Luật", "Thông tư", "Quyết định",
                 "Pháp lệnh", "Nghị quyết", "Bộ luật", "Chỉ thị"]
LOAI_PATTERN = re.compile("(" + "|".join(re.escape(x) for x in LOAI_VB_CANON) + ")", re.IGNORECASE)

def extract_vb_info(passage):
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

def chunk_passage(passage, doc_id):
    matches = list(DIEU_RE.finditer(passage))
    if not matches:
        return [{"id": f"{doc_id}_0", "dieu_so": "0", "loai_vb": "", "so_hieu": "", "text": passage.strip()}]
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(passage)
        dieu = m.group(1)
        chunks.append({
            "id": f"{doc_id}_{dieu}_{i}",
            "dieu_so": dieu,
            "loai_vb": "",
            "so_hieu": "",
            "text": passage[start:end].strip()
        })
    return chunks

def load_corpus(contexts_dir):
    if not contexts_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy {contexts_dir}")
    files = sorted(contexts_dir.glob("context_*.json"))
    if not files:
        nested = contexts_dir / "selected-contexts"
        if nested.exists():
            files = sorted(nested.glob("context_*.json"))
    if not files:
        raise FileNotFoundError("Không tìm thấy context_*.json")
    all_chunks = []
    for fp in tqdm(files, desc="Chunking"):
        try:
            with fp.open(encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        passage = doc.get("passage")
        if not passage:
            continue
        chunks = chunk_passage(passage, doc["id"])
        loai_vb, so_hieu = extract_vb_info(passage)
        for c in chunks:
            c["loai_vb"], c["so_hieu"] = loai_vb, so_hieu
        all_chunks.extend(chunks)
    print(f"  {len(files)} files -> {len(all_chunks)} chunks")
    return all_chunks

# =========================== BƯỚC 2: BM25 ===========================
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
def tokenize_simple(text):
    return _TOKEN_RE.findall(text.lower())

def build_bm25(all_chunks):
    tokenized = [tokenize_simple(f"{c.get('loai_vb','')} {c['text']}") for c in all_chunks]
    return BM25Okapi(tokenized)

# =========================== BƯỚC 3: SINH NHÃN ===========================
DIEU_CITATION_RE = re.compile(r"Điều\s+(\d+)\s*[a-zđA-ZĐ]?\b")
def extract_citations(answer):
    out = []
    for m in DIEU_CITATION_RE.finditer(answer):
        window = answer[m.end():m.end()+60]
        so_m = SO_HIEU_RE.search(window)
        if so_m and so_m.start() <= 40:
            out.append((m.group(1), so_m.group(0)))
    return out

def norm_so_hieu(s):
    return s.strip().upper()

def build_train_pairs(train_data, all_chunks):
    so_hieu_index = {}
    for c in all_chunks:
        if c["so_hieu"] and c["dieu_so"] != "0":
            key = (c["dieu_so"], norm_so_hieu(c["so_hieu"]))
            so_hieu_index.setdefault(key, c["id"])
    positive = {}
    for qid, item in train_data.items():
        for dieu, so_hieu in extract_citations(item["answer"]):
            key = (dieu, norm_so_hieu(so_hieu))
            if key in so_hieu_index:
                positive[qid] = so_hieu_index[key]
                break
    chunk_by_id = {c["id"]: c for c in all_chunks}
    return positive, chunk_by_id

# =========================== BƯỚC 4: DENSE RETRIEVER ===========================
def load_dense_model(device):
    model = SentenceTransformer(DENSE_MODEL_NAME, device=device)
    model.max_seq_length = DENSE_MAX_SEQ_LEN
    return model

def build_dense_rows(train_positive, train_data, chunk_by_id, all_chunks, bm25, n_neg=4):
    rows = []
    all_list = all_chunks
    for qid, pos_id in tqdm(train_positive.items(), desc="Dense rows"):
        q = train_data[qid]["question"]
        pos_text = chunk_by_id[pos_id]["text"]
        token_q = tokenize_simple(q)
        ranked = bm25.get_top_n(token_q, list(range(len(all_list))), n=100)
        semi_hard = ranked[10:50]
        neg_ids = [all_list[idx]["id"] for idx in semi_hard if all_list[idx]["id"] != pos_id][:n_neg]
        if len(neg_ids) < n_neg:
            pool = [c["id"] for c in all_list if c["id"] != pos_id]
            while len(neg_ids) < n_neg and pool:
                neg_ids.append(random.choice(pool))
        row = {"anchor": q, "positive": pos_text}
        for i, nid in enumerate(neg_ids[:n_neg]):
            row[f"negative_{i+1}"] = chunk_by_id[nid]["text"]
        rows.append(row)
    return rows

def finetune_dense(model, train_positive, train_data, chunk_by_id, all_chunks, bm25):
    if len(train_positive) < MIN_TRAIN_PAIRS:
        print("  Bỏ fine-tune dense (không đủ positive).")
        return model
    if len(train_positive) > MAX_TRAIN_EXAMPLES_DENSE:
        sampled = random.sample(list(train_positive.keys()), MAX_TRAIN_EXAMPLES_DENSE)
        train_positive = {qid: train_positive[qid] for qid in sampled}
        print(f"  Lấy mẫu {MAX_TRAIN_EXAMPLES_DENSE} cho dense.")
    rows = build_dense_rows(train_positive, train_data, chunk_by_id, all_chunks, bm25)
    dataset = Dataset.from_list(rows)
    print(f"  Dense training rows: {len(dataset)}")
    loss = losses.MultipleNegativesRankingLoss(model)
    batch_size = TRAIN_BATCH_SIZE_DENSE
    for attempt in range(3):
        try:
            args = SentenceTransformerTrainingArguments(
                output_dir="dense_finetuned",
                num_train_epochs=2,
                per_device_train_batch_size=batch_size,
                learning_rate=2e-5,
                warmup_ratio=0.05,
                lr_scheduler_type="cosine",
                logging_steps=20,
                save_strategy="no",
                report_to=[],
                fp16=torch.cuda.is_available(),
            )
            trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
            trainer.train()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                print(f"  [OOM] giảm batch dense: {batch_size} -> {batch_size//2}")
                batch_size = max(1, batch_size//2)
                continue
            raise
    model.save_pretrained("dense_finetuned")
    if torch.cuda.is_available():
        model = SentenceTransformer("dense_finetuned", device='cuda')
        model.max_seq_length = DENSE_MAX_SEQ_LEN
    return model

# =========================== BƯỚC 5: ENCODE CORPUS ===========================
def encode_corpus(model, all_chunks):
    if EMBED_CACHE_FILE.exists():
        with EMBED_CACHE_FILE.open("rb") as f:
            return pickle.load(f)
    if torch.cuda.is_available():
        _cap_cuda_memory(0.92)
        model = model.to('cuda').half()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Encoding on {device} (fp16)" if device == "cuda" else f"  Encoding on {device}")
    embeddings = model.encode(
        [c["text"] for c in all_chunks],
        batch_size=ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=device
    )
    with EMBED_CACHE_FILE.open("wb") as f:
        pickle.dump(embeddings, f)
    return embeddings

# =========================== BƯỚC 6: RERANKER ===========================
def load_reranker(device):
    if os.path.exists("reranker_finetuned"):
        return CrossEncoder("reranker_finetuned", device=device)
    return CrossEncoder(RERANKER_MODEL_NAME, num_labels=1, device=device)

def build_reranker_pairs(train_positive, train_data, chunk_by_id, all_chunks, bm25):
    pairs = []
    sampled = random.sample(list(train_positive.keys()), min(MAX_TRAIN_EXAMPLES_RERANKER, len(train_positive)))
    for qid in tqdm(sampled, desc="Reranker pairs"):
        q = train_data[qid]["question"]
        pos = chunk_by_id[train_positive[qid]]
        pairs.append((q, pos['text'], 1))
        token_q = tokenize_simple(q)
        ranked = bm25.get_top_n(token_q, list(range(len(all_chunks))), n=30)
        neg_ids = [all_chunks[idx]["id"] for idx in ranked if all_chunks[idx]["id"] != train_positive[qid]][:3]
        for nid in neg_ids:
            neg = chunk_by_id[nid]
            pairs.append((q, neg['text'], 0))
    random.shuffle(pairs)
    print(f"  Reranker pairs: {len(pairs)}")
    return pairs

def finetune_reranker(reranker, train_positive, train_data, chunk_by_id, all_chunks, bm25):
    if len(train_positive) < MIN_TRAIN_PAIRS:
        print("  Bỏ fine-tune reranker.")
        return reranker
    pairs = build_reranker_pairs(train_positive, train_data, chunk_by_id, all_chunks, bm25)
    batch_size = TRAIN_BATCH_SIZE_RERANKER
    for attempt in range(3):
        try:
            reranker.fit(
                train_data=pairs,
                epochs=2,
                batch_size=batch_size,
                warmup_steps=100,
                output_path="reranker_finetuned",
                show_progress_bar=True
            )
            return CrossEncoder("reranker_finetuned", device='cuda')
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                print(f"  [OOM] giảm batch reranker: {batch_size} -> {batch_size//2}")
                batch_size = max(1, batch_size//2)
                continue
            raise
    return reranker

# =========================== BƯỚC 7: RETRIEVAL + RERANK + ANSWER ===========================
def hybrid_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks,
                    reranker=None, top_k_initial=100, top_k_final=10):
    token_q = tokenize_simple(question)
    bm25_ranked = bm25.get_top_n(token_q, list(range(len(all_chunks))), n=top_k_initial)
    q_emb = dense_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    dense_scores = dense_embeddings @ q_emb
    dense_ranked = list(np.argsort(-dense_scores)[:top_k_initial])
    # RRF
    bm25_rank_map = {idx: r for r, idx in enumerate(bm25_ranked)}
    dense_rank_map = {idx: r for r, idx in enumerate(dense_ranked)}
    all_idx = set(bm25_ranked) | set(dense_ranked)
    rrf = {}
    for i in all_idx:
        r1 = bm25_rank_map.get(i, top_k_initial+1)
        r2 = dense_rank_map.get(i, top_k_initial+1)
        rrf[i] = 1/(60+r1) + 1/(60+r2)
    sorted_idx = sorted(rrf, key=rrf.get, reverse=True)[:top_k_initial]
    candidates = [all_chunks[i] for i in sorted_idx]
    if reranker is None:
        return candidates[:top_k_final]
    # Rerank
    pairs = [(question, c['text']) for c in candidates]
    scores = reranker.predict(pairs, batch_size=32)
    sorted_candidates = [c for c, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)]
    return sorted_candidates[:top_k_final]

def render_answer(chunks, top_n):
    parts, seen = [], set()
    for c in chunks:
        if c["id"] in seen or len(parts) >= top_n:
            continue
        seen.add(c["id"])
        dieu = c["dieu_so"]
        loai = c["loai_vb"] or "văn bản"
        so = c["so_hieu"] or ""
        lead = f"Theo Điều {dieu} {loai} {so} quy định cụ thể:" if dieu != "0" else f"Theo {loai} {so} quy định cụ thể:"
        parts.append(f"{lead}\n{c['text']}")
    return "\n\n".join(parts)

def answer_question(question, bm25, dense_model, dense_embeddings, all_chunks, top_n, reranker=None):
    chunks = hybrid_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks,
                             reranker, TOP_K_RETRIEVE, TOP_K_RERANK)
    if not chunks:
        return "Không tìm thấy thông tin pháp lý."
    return render_answer(chunks, top_n)

# =========================== BƯỚC 8: DEV-EVAL ===========================
def try_dev_eval(bm25, dense_model, dense_embeddings, all_chunks, train_data, reranker=None):
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
        print(f"  Bỏ qua dev-eval ({e}), dùng TOP_N=3")
        return 3
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    random.seed(42)
    ids = random.sample(list(train_data.keys()), min(300, len(train_data)))
    best_n, best_m = 3, -1.0
    for top_n in TOP_N_CANDIDATES:
        ms, rs = [], []
        for qid in tqdm(ids, desc=f"Eval top{top_n}"):
            item = train_data[qid]
            pred = answer_question(item["question"], bm25, dense_model, dense_embeddings,
                                   all_chunks, top_n, reranker)
            ref = item["answer"]
            ms.append(meteor_score([str(ref).split()], str(pred).split()))
            rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
        m, r = sum(ms)/len(ms), sum(rs)/len(rs)
        print(f"  top_n={top_n}  METEOR={m:.4f}  ROUGE-L={r:.4f}")
        if m > best_m:
            best_m, best_n = m, top_n
    print(f"  Chọn TOP_N={best_n} (METEOR={best_m:.4f})")
    return best_n

# =========================== BƯỚC 9: SUBMISSION ===========================
def build_submission(answers, expected_ids, out_zip):
    got = set(answers.keys())
    if got != expected_ids:
        raise ValueError(f"Key mismatch: missing {len(expected_ids-got)}, extra {len(got-expected_ids)}")
    for qid, ans in answers.items():
        if not isinstance(ans, str) or not ans.strip():
            print(f"  [WARN] {qid} empty")
    normalized = {qid: {"answer": str(ans)} for qid, ans in answers.items()}
    json_path = out_zip.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="submission.json")
    print(f"  OK — {out_zip}")

# =========================== MAIN ===========================
def main():
    print("=== LEGALQA ADVANCED (DENSE + RERANKER) ===")
    checkpoint("Start")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print("\n=== Bước 1: Chunk corpus ===")
    all_chunks = load_corpus(CONTEXTS_DIR)
    checkpoint("Chunk done")

    print("\n=== Bước 2: BM25 index ===")
    bm25 = build_bm25(all_chunks)
    checkpoint("BM25 done")

    print("\n=== Bước 3: Sinh nhãn từ train.json ===")
    with TRAIN_PATH.open(encoding="utf-8") as f:
        train_data = json.load(f)
    train_positive, chunk_by_id = build_train_pairs(train_data, all_chunks)
    print(f"  Positive pairs: {len(train_positive)}/{len(train_data)}")
    checkpoint("Train pairs done")

    print("\n=== Bước 4: Dense retriever ===")
    dense_model = load_dense_model(device)
    if USE_FINETUNE_DENSE and len(train_positive) >= MIN_TRAIN_PAIRS:
        dense_model = finetune_dense(dense_model, train_positive, train_data, chunk_by_id, all_chunks, bm25)
    checkpoint("Dense done")

    print("\n=== Bước 5: Encode corpus ===")
    dense_embeddings = encode_corpus(dense_model, all_chunks)
    checkpoint("Encode done")

    print("\n=== Bước 6: Reranker ===")
    reranker = load_reranker(device)
    if USE_FINETUNE_RERANKER and len(train_positive) >= MIN_TRAIN_PAIRS:
        reranker = finetune_reranker(reranker, train_positive, train_data, chunk_by_id, all_chunks, bm25)
    checkpoint("Reranker done")

    print("\n=== Bước 7: Dev-eval ===")
    top_n = try_dev_eval(bm25, dense_model, dense_embeddings, all_chunks, train_data, reranker)
    checkpoint("Dev-eval done")

    print("\n=== Bước 8: Predict public ===")
    with PUBLIC_PATH.open(encoding="utf-8") as f:
        questions = json.load(f)
    answers = {}
    for qid, item in tqdm(questions.items(), desc="Predict"):
        answers[qid] = answer_question(item["question"], bm25, dense_model, dense_embeddings,
                                       all_chunks, top_n, reranker)
    empty = sum(1 for a in answers.values() if not a.strip())
    print(f"  {len(answers)} answers, {empty} empty")
    checkpoint("Predict done")

    print("\n=== Bước 9: Đóng gói submission.zip ===")
    build_submission(answers, set(questions.keys()), OUT_ZIP)
    checkpoint("All done")

if __name__ == "__main__":
    main()