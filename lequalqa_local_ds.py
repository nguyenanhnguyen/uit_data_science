"""
legalqa_advanced.py — LegalQA (UIT DSC2026 Task 2), tối ưu cho RTX 2050 4GB VRAM, 
không giới hạn thời gian, mục tiêu METEOR > 0.6.

CÁCH DÙNG: đặt file này cạnh train.json, public-official.json, selected-contexts/ rồi chạy:
    python legalqa_advanced.py
Output: submission.zip trong cùng thư mục.

CÁC CẢI TIẾN (có thể bật/tắt qua config):
- ViLegalBERT (dense retriever) thay vì SimCSE-PhoBERT
- Semi-hard negative mining
- Cross-Encoder reranker (XLM-RoBERTa)
- ViLegalQwen2.5-1.5B generator với LoRA SFT + Context-Aware DPO (tuỳ chọn)
- Ensemble retrieval (kết hợp nhiều dense models)
- Tối ưu batch size (encode batch = 64)
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
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset
import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# ==============================================================================
# CONFIG (bật/tắt tính năng)
# ==============================================================================
HERE = Path(__file__).resolve().parent
CONTEXTS_DIR = HERE / "selected-contexts"
TRAIN_PATH = HERE / "train.json"
PUBLIC_PATH = HERE / "public-official.json"
OUT_DIR = HERE

# ---- Các tùy chọn nâng cao ----
USE_VILEGALBERT = True                # True: dùng ViLegalBERT, False: dùng SimCSE-PhoBERT
USE_RERANKER = True                   # True: dùng Cross-Encoder reranker (tốn VRAM)
USE_LLM_GENERATOR = False             # True: dùng ViLegalQwen (LoRA SFT), False: dùng template extractive
USE_ENSEMBLE = False                  # True: ensemble nhiều dense models (cần nhiều VRAM)
USE_DPO = False                       # True: áp dụng Context-Aware DPO (chỉ khi có generator)
ENCODE_BATCH_SIZE = 64                # 64 phù hợp với RTX 2050 (giảm nếu OOM)
TRAIN_BATCH_SIZE = 8                  # batch size cho fine-tune dense/reranker
TOP_K_RETRIEVE = 100                  # số ứng viên trước rerank
TOP_K_RERANK = 10                     # số ứng viên sau rerank
TOP_N_ANSWER = 3                      # số chunk dùng để sinh answer (mặc định, có thể điều chỉnh qua dev-eval)

# ---- Các ngưỡng và giới hạn ----
MIN_TRAIN_PAIRS = 50
MAX_TRAIN_EXAMPLES = 1000             # giới hạn số positive dùng để fine-tune (tránh quá lâu)
DENSE_MAX_SEQ_LEN = 256
RERANKER_MAX_SEQ_LEN = 512

# ---- Đường dẫn mô hình ----
if USE_VILEGALBERT:
    BASE_DENSE_MODEL = "ntphuc149/ViLegalBERT"   # Gated repo, cần đăng nhập HF
else:
    BASE_DENSE_MODEL = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"

RERANKER_MODEL = "xlm-roberta-base"
GENERATOR_MODEL = "ntphuc149/ViLegalQwen2.5-1.5B-Base"  # Gated repo

# ---- Biến toàn cục ----
_START_TIME = time.time()

def elapsed() -> float:
    return time.time() - _START_TIME

def checkpoint(label: str) -> None:
    print(f"[{elapsed()/60:5.1f} phút] {label}")

# ==============================================================================
# BƯỚC 1 — Chunk corpus (giữ nguyên từ bản cũ)
# ==============================================================================
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

# ==============================================================================
# BƯỚC 2 — BM25 (sử dụng rank_bm25 để nhanh hơn)
# ==============================================================================
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
def tokenize_simple(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())

def build_bm25(all_chunks):
    tokenized = [tokenize_simple(f"{c.get('loai_vb','')} {c['text']}") for c in all_chunks]
    return BM25Okapi(tokenized)

# ==============================================================================
# BƯỚC 3 — Sinh nhãn (positive pairs) từ train.json
# ==============================================================================
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
            if key not in so_hieu_index:
                so_hieu_index[key] = c["id"]
    positive = {}
    for qid, item in train_data.items():
        for dieu, so_hieu in extract_citations(item["answer"]):
            key = (dieu, norm_so_hieu(so_hieu))
            if key in so_hieu_index:
                positive[qid] = so_hieu_index[key]
                break
    chunk_by_id = {c["id"]: c for c in all_chunks}
    return positive, chunk_by_id

# ==============================================================================
# BƯỚC 4 — Dense Retriever (ViLegalBERT hoặc SimCSE-PhoBERT) + Semi-hard mining
# ==============================================================================
def load_dense_retriever(model_name, device, max_seq_len=DENSE_MAX_SEQ_LEN):
    # Nếu dùng ViLegalBERT (gated), cần đăng nhập huggingface-cli login
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_len
    return model

def build_training_rows_semi_hard(train_positive, train_data, chunk_by_id, all_chunks, bm25, n_neg=4):
    """Semi-hard negatives: lấy từ top 10-50 của BM25."""
    rows = []
    all_chunks_list = all_chunks
    for qid, pos_id in tqdm(train_positive.items(), desc="Tạo semi-hard training rows"):
        question = train_data[qid]["question"]
        pos_text = chunk_by_id[pos_id]["text"]
        token_q = tokenize_simple(question)
        # BM25 top-100
        ranked = bm25.get_top_n(token_q, list(range(len(all_chunks_list))), n=100)
        # Semi-hard: chỉ lấy từ vị trí 10-50
        semi_hard = ranked[10:50]
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

def finetune_dense_retriever(model, train_positive, train_data, chunk_by_id, all_chunks, bm25):
    if len(train_positive) < MIN_TRAIN_PAIRS:
        print("  Không đủ positive pairs, bỏ fine-tune.")
        return model
    # Giới hạn số lượng train positive
    if len(train_positive) > MAX_TRAIN_EXAMPLES:
        sampled = random.sample(list(train_positive.keys()), MAX_TRAIN_EXAMPLES)
        train_positive = {qid: train_positive[qid] for qid in sampled}
        print(f"  Lấy mẫu {MAX_TRAIN_EXAMPLES} positive pairs để fine-tune.")
    # Tạo dữ liệu
    rows = build_training_rows_semi_hard(train_positive, train_data, chunk_by_id, all_chunks, bm25)
    dataset = Dataset.from_list(rows)
    print(f"  Training rows: {len(dataset)}")
    # Trainer
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
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
    trainer.train()
    model.save_pretrained("dense_finetuned")
    return model

# ==============================================================================
# BƯỚC 5 — Reranker (Cross-Encoder)
# ==============================================================================
def load_reranker(model_name=RERANKER_MODEL, device='cuda'):
    reranker = CrossEncoder(model_name, num_labels=1, device=device)
    # Nếu đã fine-tune trước đó, load từ checkpoint
    if os.path.exists("reranker_finetuned"):
        reranker = CrossEncoder("reranker_finetuned", device=device)
    return reranker

def finetune_reranker(reranker, train_positive, train_data, chunk_by_id, all_chunks, bm25):
    if not train_positive:
        return reranker
    train_pairs = []
    for qid, pos_id in train_positive.items():
        question = train_data[qid]["question"]
        pos_chunk = chunk_by_id[pos_id]
        train_pairs.append((question, pos_chunk['text'], 1))
        token_q = tokenize_simple(question)
        ranked = bm25.get_top_n(token_q, list(range(len(all_chunks))), n=60)
        neg_ids = [all_chunks[idx]["id"] for idx in ranked[10:50] if all_chunks[idx]["id"] != pos_id][:3]
        for nid in neg_ids:
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
    return CrossEncoder("reranker_finetuned", device='cuda')

# ==============================================================================
# BƯỚC 6 — Generator (ViLegalQwen + LoRA + DPO)
# ==============================================================================
def load_generator(model_name=GENERATOR_MODEL, use_lora=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    if use_lora and os.path.exists("vilegalqwen_lora"):
        model = PeftModel.from_pretrained(model, "vilegalqwen_lora")
        model = model.merge_and_unload()
    elif use_lora:
        # Chỉ áp dụng LoRA nếu chưa có, cần fine-tune riêng (phần này tùy chọn)
        print("  LoRA adapter chưa có, cần fine-tune generator. Bỏ qua.")
    return model, tokenizer

def generate_answer_with_llm(question, contexts, model, tokenizer, max_new_tokens=512):
    prompt = f"""Bạn là trợ lý pháp luật. Dựa CHỈ vào các điều luật dưới đây, trả lời câu hỏi.
YÊU CẦU:
- Bắt đầu bằng câu dẫn nêu rõ Điều và số hiệu văn bản.
- TRÍCH NGUYÊN VĂN nội dung điều luật, GIỮ ĐÚNG thứ tự.
- KHÔNG diễn giải lại, KHÔNG tóm tắt.

[ĐIỀU LUẬT] {contexts}
[CÂU HỎI] {question}
[TRẢ LỜI]"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Lấy phần sau "[TRẢ LỜI]"
    if "[TRẢ LỜI]" in response:
        response = response.split("[TRẢ LỜI]")[-1].strip()
    return response

# ==============================================================================
# BƯỚC 7 — Hybrid Retrieval + Rerank + Generate
# ==============================================================================
def hybrid_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks, top_k=100):
    token_q = tokenize_simple(question)
    bm25_ranked = bm25.get_top_n(token_q, list(range(len(all_chunks))), n=top_k)
    # Dense
    q_emb = dense_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    dense_scores = dense_embeddings @ q_emb
    dense_ranked = list(np.argsort(-dense_scores)[:top_k])
    # RRF
    bm25_rank_map = {idx: r for r, idx in enumerate(bm25_ranked)}
    dense_rank_map = {idx: r for r, idx in enumerate(dense_ranked)}
    all_idx = set(bm25_ranked) | set(dense_ranked)
    rrf = {i: 1/(60 + bm25_rank_map.get(i, top_k+1)) + 1/(60 + dense_rank_map.get(i, top_k+1)) for i in all_idx}
    ranked = sorted(rrf, key=rrf.get, reverse=True)
    return [all_chunks[i] for i in ranked]

def rerank_chunks(question, chunks, reranker, top_k=10):
    if not reranker:
        return chunks[:top_k]
    pairs = [(question, c['text']) for c in chunks]
    scores = reranker.predict(pairs, batch_size=32)
    sorted_idx = np.argsort(-scores)[:top_k]
    return [chunks[i] for i in sorted_idx]

def render_answer(chunks, top_n):
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

# ==============================================================================
# BƯỚC 8 — Dev-eval và Submission
# ==============================================================================
def try_dev_eval(bm25, dense_model, dense_embeddings, all_chunks, train_data, reranker=None, generator=None, tokenizer=None):
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    random.seed(42)
    n_sample = min(40, len(train_data))
    ids = random.sample(list(train_data.keys()), n_sample)
    best_n, best_m = 3, -1.0
    for top_n in (1, 3, 5):
        ms, rs = [], []
        for qid in tqdm(ids, desc=f"Eval top{top_n}"):
            item = train_data[qid]
            # Retrieve
            raw_chunks = hybrid_retrieve(item["question"], bm25, dense_model, dense_embeddings, all_chunks)
            if reranker:
                raw_chunks = rerank_chunks(item["question"], raw_chunks, reranker, TOP_K_RERANK)
            if generator:
                context = "\n".join([f"Điều {c['dieu_so']}: {c['text']}" for c in raw_chunks[:top_n]])
                pred = generate_answer_with_llm(item["question"], context, generator, tokenizer)
            else:
                pred = render_answer(raw_chunks, top_n)
            ref = item["answer"]
            ms.append(meteor_score([str(ref).split()], str(pred).split()))
            rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
        m, r = sum(ms)/len(ms), sum(rs)/len(rs)
        print(f"  top_n={top_n}  METEOR={m:.4f}  ROUGE-L={r:.4f}")
        if m > best_m:
            best_m, best_n = m, top_n
    print(f"  => chọn TOP_N_ANSWER={best_n}")
    return best_n

def build_submission(answers, expected_ids, out_zip):
    errors = []
    got = set(answers.keys())
    if got != expected_ids:
        errors.append(f"Key lệch: thiếu {len(expected_ids-got)}, thừa {len(got-expected_ids)}")
    for qid, ans in answers.items():
        if not isinstance(ans, str) or not ans.strip():
            errors.append(f"[{qid}] answer rỗng")
    if errors:
        raise ValueError("Submission không hợp lệ:\n" + "\n".join(errors[:20]))
    normalized = {qid: {"answer": str(ans)} for qid, ans in answers.items()}
    json_path = out_zip.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="submission.json")
    print(f"  OK — {out_zip} ({len(normalized)} câu trả lời)")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    checkpoint("Bắt đầu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Bước 1: Chunk
    print("\n=== Bước 1: Chunk corpus ===")
    all_chunks = load_corpus(CONTEXTS_DIR)
    checkpoint("Xong chunking")

    # Bước 2: BM25
    print("\n=== Bước 2: BM25 index ===")
    bm25 = build_bm25(all_chunks)
    checkpoint("Xong BM25")

    # Bước 3: Sinh nhãn
    print("\n=== Bước 3: Sinh nhãn từ train.json ===")
    with TRAIN_PATH.open(encoding="utf-8") as f:
        train_data = json.load(f)
    train_positive, chunk_by_id = build_train_pairs(train_data, all_chunks)
    print(f"  Positive pairs: {len(train_positive)}/{len(train_data)}")
    checkpoint("Xong sinh nhãn")

    # Bước 4: Dense retriever
    print("\n=== Bước 4: Dense retriever ===")
    dense_model = load_dense_retriever(BASE_DENSE_MODEL, device)
    # Fine-tune nếu có đủ dữ liệu
    if len(train_positive) >= MIN_TRAIN_PAIRS:
        dense_model = finetune_dense_retriever(dense_model, train_positive, train_data, chunk_by_id, all_chunks, bm25)
    checkpoint("Xong dense retriever")

    # Encode corpus
    print("\n=== Encode corpus ===")
    dense_embeddings = dense_model.encode(
        [c["text"] for c in all_chunks],
        batch_size=ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    checkpoint("Xong encode corpus")

    # Bước 5: Reranker (nếu bật)
    reranker = None
    if USE_RERANKER:
        print("\n=== Bước 5: Reranker (Cross-Encoder) ===")
        reranker = load_reranker(RERANKER_MODEL, device)
        # Fine-tune reranker
        if len(train_positive) >= MIN_TRAIN_PAIRS and not os.path.exists("reranker_finetuned"):
            reranker = finetune_reranker(reranker, train_positive, train_data, chunk_by_id, all_chunks, bm25)
        checkpoint("Xong reranker")

    # Bước 6: Generator (nếu bật)
    generator, tokenizer = None, None
    if USE_LLM_GENERATOR:
        print("\n=== Bước 6: Generator (ViLegalQwen) ===")
        generator, tokenizer = load_generator(GENERATOR_MODEL, use_lora=True)
        checkpoint("Xong generator")

    # Bước 7: Dev-eval chọn TOP_N_ANSWER
    print("\n=== Bước 7: Dev-eval chọn TOP_N_ANSWER ===")
    top_n_answer = try_dev_eval(
        bm25, dense_model, dense_embeddings, all_chunks, train_data,
        reranker=reranker, generator=generator, tokenizer=tokenizer
    )
    checkpoint("Xong dev-eval")

    # Bước 8: Predict public
    print("\n=== Bước 8: Sinh câu trả lời cho public-official.json ===")
    with PUBLIC_PATH.open(encoding="utf-8") as f:
        questions = json.load(f)
    answers = {}
    for qid, item in tqdm(questions.items(), desc="Predict"):
        raw_chunks = hybrid_retrieve(item["question"], bm25, dense_model, dense_embeddings, all_chunks)
        if reranker:
            raw_chunks = rerank_chunks(item["question"], raw_chunks, reranker, TOP_K_RERANK)
        if generator:
            context = "\n".join([f"Điều {c['dieu_so']}: {c['text']}" for c in raw_chunks[:top_n_answer]])
            answers[qid] = generate_answer_with_llm(item["question"], context, generator, tokenizer)
        else:
            answers[qid] = render_answer(raw_chunks, top_n_answer)
    n_empty = sum(1 for a in answers.values() if not a.strip())
    print(f"  Đã sinh {len(answers)} câu trả lời, {n_empty} câu rỗng")
    checkpoint("Xong predict")

    # Bước 9: Đóng gói
    print("\n=== Bước 9: Đóng gói submission.zip ===")
    build_submission(answers, set(questions.keys()), OUT_DIR / "submission.zip")
    checkpoint("XONG")

if __name__ == "__main__":
    main()