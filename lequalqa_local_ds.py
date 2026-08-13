"""
legalqa_optimized.py — LegalQA (UIT DSC2026 Task 2)
Tối ưu cho RTX 2050 4GB VRAM + 32GB RAM, không generator, chỉ dense + reranker.
Mục tiêu: METEOR > 0.45, thời gian ~6-8 giờ (có cache).
Output: submission.zip
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

# =========================== CONFIG ===========================
HERE = Path(__file__).resolve().parent
CONTEXTS_DIR = HERE / "selected-contexts"
TRAIN_PATH = HERE / "train.json"
PUBLIC_PATH = HERE / "public-official.json"

DENSE_MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"  # 135M
RERANKER_MODEL_NAME = "xlm-roberta-base"                           # 278M

# ---- Hyperparameters (tối ưu cho tốc độ) ----
DENSE_MAX_SEQ_LEN = 128          # Giảm để encode nhanh hơn
RERANKER_MAX_SEQ_LEN = 512
ENCODE_BATCH_SIZE = 128          # Tăng batch size
TRAIN_BATCH_SIZE = 8
TOP_K_RETRIEVE = 100
TOP_K_RERANK = 10
TOP_N_ANSWER = 3

# ---- Fine-tune options (bạn có thể tắt nếu đã có model fine-tuned) ----
USE_FINETUNE_DENSE = True        # Tắt nếu muốn chạy nhanh hơn (dùng zero-shot)
USE_FINETUNE_RERANKER = True     # Tắt nếu muốn nhanh
MIN_TRAIN_PAIRS = 50
MAX_TRAIN_EXAMPLES = 1000

# ---- Cache ----
EMBED_CACHE_FILE = HERE / "corpus_embeddings.pkl"
OUT_ZIP = HERE / "submission.zip"

# =========================== UTILITIES ===========================
_START_TIME = time.time()
def elapsed() -> float:
    return time.time() - _START_TIME

def checkpoint(label: str) -> None:
    print(f"[{elapsed()/60:5.1f} phút] {label}")

def move_to_device(model, device):
    """Ép model lên device và in log."""
    if hasattr(model, 'to'):
        model = model.to(device)
        print(f"  Model moved to {device}")
    return model

# =========================== BƯỚC 1: CHUNK CORPUS ===========================
DIEU_RE = re.compile(r"^[ \t]*Điều\s+(\d+)[a-zđA-ZĐ]?[\.\s]", re.MULTILINE)
SO_HEADER_RE = re.compile(r"Số\s*[:：]\s*([0-9A-Za-zĐđ/\-]+)")
SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zĐđ]{2,10}(?:-[A-Za-zĐđ]{2,10})?")
LOAI_VB_CANON = ["Thông tư liên tịch", "Nghị định", "Luật", "Thông tư", "Quyết định",
                 "Pháp lệnh", "Nghị quyết", "Bộ luật", "Chỉ thị"]
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
    matches = list(DIEU_RE.finditer(passage))
    if not matches:
        return [{"id": f"{doc_id}_0", "dieu_so": "0", "loai_vb": "", "so_hieu": "", "text": passage.strip()}]
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(passage)
        dieu = m.group(1)
        chunks.append({"id": f"{doc_id}_{dieu}_{i}", "dieu_so": dieu, "loai_vb": "", "so_hieu": "",
                        "text": passage[start:end].strip()})
    return chunks

def load_corpus(contexts_dir: Path) -> list:
    if not contexts_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy {contexts_dir}")
    files = sorted(contexts_dir.glob("context_*.json"))
    if not files:
        nested = contexts_dir / "selected-contexts"
        if nested.exists():
            files = sorted(nested.glob("context_*.json"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy context_*.json trong {contexts_dir}")
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
    print(f"  {len(files)} văn bản -> {len(all_chunks)} chunk.")
    return all_chunks

# =========================== BƯỚC 2: BM25 ===========================
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
def tokenize_simple(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())

def build_bm25(all_chunks):
    tokenized = [tokenize_simple(f"{c.get('loai_vb','')} {c['text']}") for c in all_chunks]
    return BM25Okapi(tokenized)

# =========================== BƯỚC 3: SINH NHÃN ===========================
DIEU_CITATION_RE = re.compile(r"Điều\s+(\d+)\s*[a-zđA-ZĐ]?\b")

def extract_citations(answer: str) -> list:
    out = []
    for m in DIEU_CITATION_RE.finditer(answer):
        window = answer[m.end(): m.end() + 60]
        so_m = SO_HIEU_RE.search(window)
        if so_m and so_m.start() <= 40:
            out.append((m.group(1), so_m.group(0)))
    return out

def norm_so_hieu(s: str) -> str:
    return s.strip().upper()

def build_train_pairs(train_data: dict, all_chunks: list):
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

# =========================== BƯỚC 4: DENSE RETRIEVER (có ép GPU) ===========================
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset

def load_dense_model(device='cuda'):
    model = SentenceTransformer(DENSE_MODEL_NAME, device=device)
    model.max_seq_length = DENSE_MAX_SEQ_LEN
    # Ép model lên device rõ ràng
    if device == 'cuda':
        model = move_to_device(model, device)
    print(f"  Dense model device: {model.device}")
    return model

def build_semi_hard_rows(train_positive, train_data, chunk_by_id, all_chunks, bm25, n_neg=4):
    rows = []
    all_chunks_list = all_chunks
    for qid, pos_id in tqdm(train_positive.items(), desc="Tạo semi-hard training rows"):
        question = train_data[qid]["question"]
        pos_text = chunk_by_id[pos_id]["text"]
        token_q = tokenize_simple(question)
        ranked = bm25.get_top_n(token_q, list(range(len(all_chunks_list))), n=100)
        semi_hard = ranked[10:50]  # semi-hard: từ vị trí 10 đến 50
        neg_ids = [all_chunks_list[idx]["id"] for idx in semi_hard if all_chunks_list[idx]["id"] != pos_id][:n_neg]
        if len(neg_ids) < n_neg:
            pool = [c["id"] for c in all_chunks_list if c["id"] != pos_id]
            while len(neg_ids) < n_neg and pool:
                neg_ids.append(random.choice(pool))
        row = {"anchor": question, "positive": pos_text}
        for i, nid in enumerate(neg_ids[:n_neg]):
            row[f"negative_{i+1}"] = chunk_by_id[nid]["text"]
        rows.append(row)
    return rows

def finetune_dense(model, train_positive, train_data, chunk_by_id, all_chunks, bm25):
    if len(train_positive) < MIN_TRAIN_PAIRS:
        print("  Không đủ positive pairs, bỏ fine-tune dense.")
        return model
    if len(train_positive) > MAX_TRAIN_EXAMPLES:
        sampled = random.sample(list(train_positive.keys()), MAX_TRAIN_EXAMPLES)
        train_positive = {qid: train_positive[qid] for qid in sampled}
        print(f"  Lấy mẫu {MAX_TRAIN_EXAMPLES} positive pairs.")
    rows = build_semi_hard_rows(train_positive, train_data, chunk_by_id, all_chunks, bm25)
    dataset = Dataset.from_list(rows)
    print(f"  Dense training rows: {len(dataset)}")
    loss = losses.MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir="dense_finetuned",
        num_train_epochs=2,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        # KHÔNG dùng no_cuda – trainer tự dùng GPU nếu có
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
    trainer.train()
    model.save_pretrained("dense_finetuned")
    # Sau khi save, nếu có GPU, load lại model và đưa về GPU
    if torch.cuda.is_available():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("dense_finetuned", device='cuda')
        model.max_seq_length = DENSE_MAX_SEQ_LEN
    return model

# =========================== BƯỚC 5: RERANKER ===========================
from sentence_transformers import CrossEncoder

def load_reranker(device='cuda'):
    if os.path.exists("reranker_finetuned"):
        print("  Load reranker from checkpoint.")
        reranker = CrossEncoder("reranker_finetuned", device=device)
    else:
        reranker = CrossEncoder(RERANKER_MODEL_NAME, num_labels=1, device=device)
    if device == 'cuda':
        reranker = move_to_device(reranker, device)
    return reranker

def finetune_reranker(reranker, train_positive, train_data, chunk_by_id, all_chunks, bm25):
    if len(train_positive) < MIN_TRAIN_PAIRS:
        print("  Không đủ positive pairs, bỏ fine-tune reranker.")
        return reranker
    train_pairs = []
    for qid, pos_id in tqdm(train_positive.items(), desc="Tạo reranker pairs"):
        question = train_data[qid]["question"]
        pos_chunk = chunk_by_id[pos_id]
        train_pairs.append((question, pos_chunk['text'], 1))
        token_q = tokenize_simple(question)
        ranked = bm25.get_top_n(token_q, list(range(len(all_chunks))), n=60)
        semi_hard = [all_chunks[idx]["id"] for idx in ranked[10:50] if all_chunks[idx]["id"] != pos_id][:3]
        for nid in semi_hard:
            neg_chunk = chunk_by_id[nid]
            train_pairs.append((question, neg_chunk['text'], 0))
    print(f"  Reranker training pairs: {len(train_pairs)}")
    random.shuffle(train_pairs)
    reranker.fit(
        train_data=train_pairs,
        epochs=2,
        batch_size=8,
        warmup_steps=100,
        output_path="reranker_finetuned",
        show_progress_bar=True
    )
    reranker = CrossEncoder("reranker_finetuned", device='cuda')
    return reranker

# =========================== BƯỚC 6: RETRIEVAL + RERANK + ANSWER ===========================
def hybrid_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks, top_k=TOP_K_RETRIEVE):
    token_q = tokenize_simple(question)
    bm25_ranked = bm25.get_top_n(token_q, list(range(len(all_chunks))), n=top_k)
    # Ép model encode trên đúng device
    q_emb = dense_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    dense_scores = dense_embeddings @ q_emb
    dense_ranked = list(np.argsort(-dense_scores)[:top_k])
    bm25_rank_map = {idx: r for r, idx in enumerate(bm25_ranked)}
    dense_rank_map = {idx: r for r, idx in enumerate(dense_ranked)}
    all_idx = set(bm25_ranked) | set(dense_ranked)
    rrf = {}
    for i in all_idx:
        r1 = bm25_rank_map.get(i, top_k+1)
        r2 = dense_rank_map.get(i, top_k+1)
        rrf[i] = 1/(60+r1) + 1/(60+r2)
    ranked = sorted(rrf, key=rrf.get, reverse=True)
    return [all_chunks[i] for i in ranked]

def rerank_chunks(question, chunks, reranker, top_k=TOP_K_RERANK):
    if not reranker:
        return chunks[:top_k]
    pairs = [(question, c['text']) for c in chunks]
    scores = reranker.predict(pairs, batch_size=32)
    sorted_idx = np.argsort(-scores)[:top_k]
    return [chunks[i] for i in sorted_idx]

def render_answer(chunks, top_n=TOP_N_ANSWER):
    parts, seen = [], set()
    for c in chunks:
        if c["id"] in seen or len(parts) >= top_n:
            continue
        seen.add(c["id"])
        loai_vb = c["loai_vb"] or "văn bản"
        so_hieu = c["so_hieu"] or ""
        dieu = c["dieu_so"]
        lead = f"Theo Điều {dieu} {loai_vb} {so_hieu} quy định cụ thể:" if dieu != "0" else f"Theo {loai_vb} {so_hieu} quy định cụ thể:"
        parts.append(f"{lead}\n{c['text']}")
    return "\n\n".join(parts)

def answer_question(question, bm25, dense_model, dense_embeddings, all_chunks, reranker=None):
    raw = hybrid_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks)
    if not raw:
        return "Không tìm thấy thông tin pháp lý."
    reranked = rerank_chunks(question, raw, reranker, TOP_K_RERANK)
    return render_answer(reranked, TOP_N_ANSWER)

# =========================== BƯỚC 7: DEV-EVAL ===========================
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
        print(f"  Bỏ qua dev-eval ({e}), dùng TOP_N_ANSWER=3.")
        return 3
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    random.seed(42)
    n_sample = min(40, len(train_data))
    ids = random.sample(list(train_data.keys()), n_sample)
    best_n, best_m = 3, -1.0
    for top_n in (1, 3, 5):
        ms, rs = [], []
        for qid in tqdm(ids, desc=f"Eval top{top_n}"):
            item = train_data[qid]
            raw = hybrid_retrieve(item["question"], bm25, dense_model, dense_embeddings, all_chunks)
            if reranker:
                raw = rerank_chunks(item["question"], raw, reranker, TOP_K_RERANK)
            pred = render_answer(raw, top_n)
            ref = item["answer"]
            ms.append(meteor_score([str(ref).split()], str(pred).split()))
            rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
        m, r = sum(ms)/len(ms), sum(rs)/len(rs)
        print(f"  top_n={top_n}  METEOR={m:.4f}  ROUGE-L={r:.4f}")
        if m > best_m:
            best_m, best_n = m, top_n
    print(f"  => chọn TOP_N_ANSWER={best_n}")
    return best_n

# =========================== BƯỚC 8: SUBMISSION ===========================
def build_submission(answers, expected_ids, out_zip):
    got = set(answers.keys())
    if got != expected_ids:
        raise ValueError(f"Key lệch: thiếu {len(expected_ids-got)}, thừa {len(got-expected_ids)}")
    for qid, ans in answers.items():
        if not isinstance(ans, str) or not ans.strip():
            print(f"  [WARN] {qid} answer rỗng")
    normalized = {qid: {"answer": str(ans)} for qid, ans in answers.items()}
    json_path = out_zip.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="submission.json")
    print(f"  OK — {out_zip} ({len(normalized)} câu trả lời)")

# =========================== MAIN ===========================
def main():
    print("=== LEGALQA OPTIMIZED PIPELINE ===")
    checkpoint("Bắt đầu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # 1. Chunk
    print("\n=== Bước 1: Chunk corpus ===")
    all_chunks = load_corpus(CONTEXTS_DIR)
    checkpoint("Xong chunking")

    # 2. BM25
    print("\n=== Bước 2: BM25 index ===")
    bm25 = build_bm25(all_chunks)
    checkpoint("Xong BM25")

    # 3. Sinh nhãn
    print("\n=== Bước 3: Sinh nhãn từ train.json ===")
    with TRAIN_PATH.open(encoding="utf-8") as f:
        train_data = json.load(f)
    train_positive, chunk_by_id = build_train_pairs(train_data, all_chunks)
    print(f"  Positive pairs: {len(train_positive)}/{len(train_data)}")
    checkpoint("Xong sinh nhãn")

    # 4. Dense Retriever
    print("\n=== Bước 4: Dense retriever ===")
    dense_model = load_dense_model(device)
    if USE_FINETUNE_DENSE and len(train_positive) >= MIN_TRAIN_PAIRS:
        dense_model = finetune_dense(dense_model, train_positive, train_data, chunk_by_id, all_chunks, bm25)
        # Sau khi fine-tune, đảm bảo model trên GPU
        if device == 'cuda':
            dense_model = move_to_device(dense_model, device)
    checkpoint("Xong dense retriever")

    # 5. Encode corpus (có cache)
    print("\n=== Encode corpus ===")
    if EMBED_CACHE_FILE.exists():
        with EMBED_CACHE_FILE.open("rb") as f:
            dense_embeddings = pickle.load(f)
        print(f"  Loaded embeddings from cache ({len(dense_embeddings)} vectors).")
    else:
        # Ép model về GPU trước khi encode (phòng trường hợp bị rơi xuống CPU)
        if device == 'cuda':
            dense_model = move_to_device(dense_model, device)
        print(f"  Encoding with batch_size={ENCODE_BATCH_SIZE} on {device}...")
        dense_embeddings = dense_model.encode(
            [c["text"] for c in all_chunks],
            batch_size=ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=device  # Truyền device rõ ràng
        )
        with EMBED_CACHE_FILE.open("wb") as f:
            pickle.dump(dense_embeddings, f)
        print(f"  Saved embeddings to {EMBED_CACHE_FILE}.")
    checkpoint("Xong encode corpus")

    # 6. Reranker
    print("\n=== Bước 5: Reranker (Cross-Encoder) ===")
    reranker = load_reranker(device)
    if USE_FINETUNE_RERANKER and len(train_positive) >= MIN_TRAIN_PAIRS:
        reranker = finetune_reranker(reranker, train_positive, train_data, chunk_by_id, all_chunks, bm25)
        if device == 'cuda':
            reranker = move_to_device(reranker, device)
    checkpoint("Xong reranker")

    # 7. Dev-eval
    print("\n=== Bước 6: Dev-eval ===")
    top_n = try_dev_eval(bm25, dense_model, dense_embeddings, all_chunks, train_data, reranker)
    global TOP_N_ANSWER
    TOP_N_ANSWER = top_n

    def render_answer_updated(chunks, top_n=TOP_N_ANSWER):
        parts, seen = [], set()
        for c in chunks:
            if c["id"] in seen or len(parts) >= top_n:
                continue
            seen.add(c["id"])
            loai_vb = c["loai_vb"] or "văn bản"
            so_hieu = c["so_hieu"] or ""
            dieu = c["dieu_so"]
            lead = f"Theo Điều {dieu} {loai_vb} {so_hieu} quy định cụ thể:" if dieu != "0" else f"Theo {loai_vb} {so_hieu} quy định cụ thể:"
            parts.append(f"{lead}\n{c['text']}")
        return "\n\n".join(parts)
    checkpoint("Xong dev-eval")

    # 8. Predict public
    print("\n=== Bước 7: Sinh câu trả lời cho public-official.json ===")
    with PUBLIC_PATH.open(encoding="utf-8") as f:
        questions = json.load(f)
    answers = {}
    for qid, item in tqdm(questions.items(), desc="Predict"):
        raw = hybrid_retrieve(item["question"], bm25, dense_model, dense_embeddings, all_chunks)
        if reranker:
            raw = rerank_chunks(item["question"], raw, reranker, TOP_K_RERANK)
        answers[qid] = render_answer_updated(raw, TOP_N_ANSWER)
    n_empty = sum(1 for a in answers.values() if not a.strip())
    print(f"  Đã sinh {len(answers)} câu trả lời, {n_empty} câu rỗng")
    checkpoint("Xong predict")

    # 9. Đóng gói
    print("\n=== Bước 8: Đóng gói submission.zip ===")
    build_submission(answers, set(questions.keys()), OUT_ZIP)
    checkpoint("XONG")

if __name__ == "__main__":
    main()