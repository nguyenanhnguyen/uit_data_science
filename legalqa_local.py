"""
legalqa_local.py — LegalQA (UIT DSC2026 Task 2), tối ưu cho RTX 2050 4GB VRAM.

CÁCH DÙNG: đặt file này cạnh train.json, public-official.json, selected-contexts/ (đúng
layout thư mục của bạn) rồi chạy:
    python legalqa_local.py
Output: submission.zip trong cùng thư mục. Mọi cache/model tải về nằm gọn trong
thư mục con `cache/` cạnh script (xem BẢN SỬA #5 bên dưới) — không rải ra chỗ khác,
không lưu checkpoint fine-tune.

THƯ VIỆN CẦN CÀI (trong venv "env" của bạn):
    pip install numpy sentence-transformers datasets "accelerate>=1.1.0" nltk rouge_score tiktoken sentencepiece
Và BẮT BUỘC kiểm tra torch có nhận đúng GPU không TRƯỚC khi chạy (chạy thử):
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only')"
Nếu ra False dù máy có RTX 2050 — cài lại torch bản có CUDA (driver mới thì dùng cu130,
driver cũ hơn thì cu126), KHÔNG dùng `pip install torch` trần trên Windows (mặc định tải
bản CPU-only).

===============================================================================
BẢN SỬA #5 (bỏ trần VRAM nhân tạo + bỏ lưu checkpoint + gom cache về một chỗ)
===============================================================================
Ba thay đổi theo yêu cầu, trên nền bản #4 đã chạy thành công (log thật, không còn bug
device CPU/GPU của bản #4 nữa — giữ nguyên mọi phần đó):

1. BỎ `_cap_cuda_memory()` (ép trần 92% VRAM bằng `set_per_process_memory_fraction`).
   Bản #4 giải thích đây là để buộc CUDA raise OutOfMemoryError sạch thay vì âm thầm
   tràn sang shared system memory (chậm hơn nhiều trên Windows WDDM) — đúng về mặt kỹ
   thuật, nhưng vẫn là một dạng ĐẶT TRẦN TRƯỚC khi biết có cần hay không. Bỏ hẳn: dùng
   thẳng 100% VRAM vật lý, chỉ dựa vào các vòng lặp OOM-backoff đã có sẵn (encode_corpus,
   finetune_or_load_dense) để lùi batch khi THẬT SỰ hết bộ nhớ — không đặt giới hạn trước.
   Vẫn giữ TF32 + cudnn.benchmark (đây là tăng tốc thật, không phải giới hạn).

2. BỎ lưu checkpoint: bản #4 gọi `model.save_pretrained("dense_finetuned")` ở cuối
   Bước 4, tạo thêm 2 thư mục (`dense_finetuned/`, `dense_finetuned_tmp/` từ output_dir
   của Trainer). Model fine-tune xong dùng NGAY cho Bước 5 trong cùng lần chạy — không
   cần persist ra đĩa. Bỏ dòng save; output_dir của Trainer trỏ vào `cache/trainer_tmp`
   (đã có `save_strategy="no"` nên không sinh file checkpoint thật, chỉ trạng thái
   tạm của Trainer).

3. GOM CACHE VỀ MỘT THƯ MỤC: bản #4 không pin HF_HOME nên model tải về rơi vào
   `~/.cache/huggingface` mặc định của hệ thống (ẩn, khó dọn), nltk tải wordnet vào vị
   trí mặc định của nltk. Từ bản này: mọi thứ tải về/cache nằm trong `cache/` cạnh
   script — `cache/hf` (HF_HOME), `cache/nltk_data`, `cache/trainer_tmp` — MỘT thư mục
   gốc, tên thống nhất, dễ dọn (`rm -rf cache/` là sạch hoàn toàn, không sợ xoá nhầm chỗ
   khác của hệ thống).

Hệ quả của (1): batch khởi điểm (ENCODE_BATCH_SIZE, TRAIN_MINI_BATCH_SIZE) được nâng lên
so với bản #4 — trước đây phải dè dặt vì đã tự trừ hao 8% VRAM, giờ dùng thẳng 100% nên
bắt đầu ở mức cao hơn cũng an toàn (OOM-backoff tự lùi nếu vẫn quá lớn, chỉ tốn 1 lần thử
hụt chứ không mất gì khác).

===============================================================================
QUYẾT ĐỊNH THIẾT KẾ CÒN GIỮ NGUYÊN CHO RÀNG BUỘC 4GB VRAM
===============================================================================
1. KHÔNG fine-tune/chạy LLM sinh câu trả lời (LoRA SFT/DPO) — 1.5B+ tham số dù QLoRA vẫn
   rủi ro OOM trên 4GB với answer luật dài. Dùng **template extractive** (ghép nguyên văn)
   — đúng phân tích công thức METEOR (alpha=0.9 nặng recall, phạt phân mảnh mũ 3).
2. Reranker fine-tune riêng: CHƯA thêm ở bản này — nếu sau khi chạy lại vẫn chưa qua 0.6,
   đây là lever tiếp theo đáng thử (đã có sẵn code mẫu ở phiên bản Kaggle trước đó).
3. Dense retriever fine-tune time-boxed dựa trên tốc độ ĐO THẬT (không phải ước lượng) —
   với GPU chạy đúng, số step trong ngân sách sẽ cao hơn nhiều so với lần chạy lỗi trước.
4. BM25 tự viết bằng numpy, chạy CPU, không tốn VRAM.
===============================================================================
"""
from __future__ import annotations
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent  # script chạy cùng thư mục với train.json, v.v.

# ---- BẢN SỬA #5: một thư mục cache duy nhất, thống nhất tên, dựng TRƯỚC mọi import
#      có thể tải mạng (transformers/sentence-transformers/nltk) để chắc chắn không cái
#      nào rơi về vị trí mặc định của hệ thống. -------------------------------------
CACHE_DIR = HERE / "cache"
HF_CACHE_DIR = CACHE_DIR / "hf"
NLTK_CACHE_DIR = CACHE_DIR / "nltk_data"
TRAINER_TMP_DIR = CACHE_DIR / "trainer_tmp"
for _d in (HF_CACHE_DIR, NLTK_CACHE_DIR, TRAINER_TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))

# SỬA (log lần 4): treo với CPU~8%/RAM ổn định = đang CHỜ MẠNG, không phải đang tính toán —
# log trước đó có "sending unauthenticated requests to HF Hub" xác nhận có gọi mạng ở bước
# này. Tắt TELEMETRY (không cần, chỉ là thống kê ẩn danh) nhưng KHÔNG tắt hẳn khả năng gọi
# mạng — xem SỬA NGHIÊM TRỌNG bên dưới.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# SỬA NGHIÊM TRỌNG (log thật: reranker AITeamVN/Vietnamese_Reranker load lỗi "couldn't connect
# ... and couldn't find them in cached files"): dòng `HF_HUB_OFFLINE=1` ở bản trước là NGUYÊN
# NHÂN THẬT của lỗi này, KHÔNG PHẢI do wifi chập chờn như tưởng lúc đó. Đặt OFFLINE=1 chặn
# TOÀN BỘ cuộc gọi mạng HuggingFace — kể cả lần ĐẦU TIÊN tải 1 model MỚI (reranker) chưa từng
# có trong cache, dù model retriever cũ đã cache sẵn. Bỏ hẳn dòng này — HF Hub tự dùng cache
# cục bộ khi có sẵn, không cần ép OFFLINE mới dùng được cache.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # tránh treo do fork trên Windows
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
import re
import json
import math
import time
import random
import zipfile
from collections import defaultdict, Counter

import numpy as np

# ==============================================================================
# CONFIG
# ==============================================================================
CONTEXTS_DIR = HERE / "selected-contexts"
TRAIN_PATH = HERE / "train.json"
PUBLIC_PATH = HERE / "public-official.json"
OUT_DIR = HERE

TIME_BUDGET_SEC = 3 * 3600         # trần an toàn tổng thể (3 giờ), chỉ để log "còn lại bao
                                    # nhiêu", KHÔNG cắt ngang dev-eval/predict giữa chừng.
FINETUNE_TIME_BUDGET_SEC = 90 * 60  # tối đa dành cho fine-tune dense retriever (Bước 4).
MIN_TRAIN_PAIRS = 50               # dưới ngưỡng này -> bỏ fine-tune, dùng zero-shot
MAX_TRAIN_EXAMPLES = 3000          # gần hết 3565 positive pairs thật.

BASE_DENSE_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
# Dùng trực tiếp cho bài toán Vietnamese Legal QA retrieval trong nghiên cứu thực tế
# (Pham et al., "Vietnamese Legal Information Retrieval in Question-Answering System",
# arXiv:2409.13699) — cùng domain với bài thi này.
DENSE_MAX_SEQ_LEN = 256            # cắt ngắn để tiết kiệm VRAM + thời gian (câu luật dài,
                                    # nhưng embedding chỉ cần đủ để phân biệt ngữ nghĩa, không
                                    # cần đọc hết toàn văn — sinh câu trả lời vẫn dùng text đầy đủ)
TRAIN_BATCH_SIZE = 32              # batch HIỆU DỤNG (số in-batch negative, quyết định CHẤT
                                    # LƯỢNG train) — tách biệt khỏi VRAM thật nhờ
                                    # CachedMultipleNegativesRankingLoss, xem TRAIN_MINI_BATCH_SIZE.
TRAIN_MINI_BATCH_SIZE = 4           # SỬA (bản #5): nâng từ 2 lên 4 — không còn trần VRAM nhân
                                    # tạo (_cap_cuda_memory đã bỏ) nên bắt đầu cao hơn cũng an
                                    # toàn, OOM-backoff bên dưới tự lùi nếu vẫn quá lớn.
N_NEG_PER_ROW = 2                   # in-batch negative từ batch lớn đã đủ, không cần nhiều
                                    # hard-negative tường minh nữa.
ENCODE_BATCH_SIZE = 64             # SỬA (bản #5): nâng từ 32 lên 64 — cùng lý do trên.

TOP_K_RETRIEVE = 100                # số ứng viên lấy ra sau RRF fusion
DEV_EVAL_SAMPLE_SIZE = 300          # mẫu lớn hơn cho tín hiệu METEOR đáng tin khi chọn TOP_N_ANSWER.

_START_TIME = time.time()


def elapsed() -> float:
    return time.time() - _START_TIME


def remaining() -> float:
    return TIME_BUDGET_SEC - elapsed()


def checkpoint(label: str) -> None:
    print(f"[{elapsed()/60:5.1f} phút] {label}  (còn lại ~{remaining()/60:.1f} phút trong ngân sách)")


# ==============================================================================
# BƯỚC 1 — Chunk corpus theo Điều (neo đầu dòng — tránh lỗi rách nội dung khi 1 Điều
# trích dẫn Điều khác trong thân bài) + trích so_hieu/loai_vb từ NỘI DUNG (không phải tên file)
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
        end = matches[i + 1].start() if i + 1 < len(matches) else len(passage)
        dieu = m.group(1)
        chunks.append({"id": f"{doc_id}_{dieu}_{i}", "dieu_so": dieu, "loai_vb": "", "so_hieu": "",
                        "text": passage[start:end].strip()})
    return chunks


def load_corpus(contexts_dir: Path) -> list:
    if not contexts_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy {contexts_dir} — kiểm tra lại layout thư mục.")
    files = sorted(contexts_dir.glob("context_*.json"))
    if not files:
        nested = contexts_dir / "selected-contexts"
        if nested.exists():
            files = sorted(nested.glob("context_*.json"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy context_*.json trong {contexts_dir}")

    all_chunks, n_no_dieu = [], 0
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
        if len(chunks) == 1 and chunks[0]["dieu_so"] == "0":
            n_no_dieu += 1
        loai_vb, so_hieu = extract_vb_info(passage)
        for c in chunks:
            c["loai_vb"], c["so_hieu"] = loai_vb, so_hieu
        all_chunks.extend(chunks)

    pct = round(100 * (1 - n_no_dieu / len(files)), 2) if files else 0.0
    print(f"  {len(files)} văn bản -> {len(all_chunks)} chunk. {pct}% có cấu trúc Điều.")
    if pct < 95.0:
        print("  [CẢNH BÁO] < 95% — kiểm tra vài context_*.json thật, có thể cần chỉnh DIEU_RE.")
    return all_chunks


# ==============================================================================
# BƯỚC 2 — BM25 tự viết bằng numpy (inverted index — nhanh, không phụ thuộc rank_bm25)
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

        # Posting list lưu bằng 2 mảng numpy (doc_idx, freq) — get_scores() dùng fancy-
        # indexing vector hoá thay vì vòng lặp Python, cần thiết cho corpus ~160k chunk.
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
            # fancy-index += đúng vì idxs KHÔNG có phần tử trùng lặp trong 1 posting list (mỗi
            # văn bản chỉ xuất hiện tối đa 1 lần trong posting của 1 từ, do build từ Counter).
            scores[idxs] += contrib
        return scores

    def top_k(self, query_tokens, k: int) -> list:
        scores = self.get_scores(query_tokens)
        return list(np.argsort(-scores)[:k])


# ==============================================================================
# BƯỚC 3 — Sinh nhãn (question -> chunk) từ citation trong train.json — dùng làm dữ liệu
# fine-tune dense retriever (Bước 4). Trích so_hieu trong CỬA SỔ ngay sau "Điều X" (không
# dùng lazy free-text — dễ bị cắt cụt ở khoảng trắng đầu tiên).
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
            so_hieu_index.setdefault((c["dieu_so"], norm_so_hieu(c["so_hieu"])), c["id"])

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
# BƯỚC 4 — Fine-tune dense retriever, TIME-BOXED (đo tốc độ vài step đầu, tự tính số step
# tối đa vừa ngân sách còn lại — KHÔNG train "epochs=N" mù quáng có thể vượt giờ).
# Có fallback: nếu CUDA OOM ở batch_size hiện tại, tự giảm 1 nửa rồi thử lại.
# ==============================================================================
def _build_training_rows(train_positive, train_data, chunk_by_id, all_chunks, bm25, n_neg: int = N_NEG_PER_ROW):
    """Trả về list dict {anchor, positive, negative_1..negative_n} — đúng format
    SentenceTransformerTrainer + MultipleNegativesRankingLoss chấp nhận trực tiếp.
    In progress mỗi 500 câu — vòng lặp này có thể chạy vài nghìn lần (mỗi lần 1 lượt BM25
    top_k trên toàn corpus), không in gì cho tới khi xong hẳn sẽ trông giống "treo" dù
    thực ra đang chạy bình thường, không phân biệt được với treo thật."""
    rows = []
    n = len(train_positive)
    for i, (qid, pos_id) in enumerate(train_positive.items()):
        question = train_data[qid]["question"]
        pos_text = chunk_by_id[pos_id]["text"]
        token_q = tokenize_simple(question)
        ranked = bm25.top_k(token_q, 60)
        neg_ids = [all_chunks[i2]["id"] for i2 in ranked[5:60] if all_chunks[i2]["id"] != pos_id][:n_neg]
        if len(neg_ids) < n_neg:
            pool = [c["id"] for c in all_chunks if c["id"] != pos_id]
            while len(neg_ids) < n_neg and pool:
                neg_ids.append(random.choice(pool))
        row = {"anchor": question, "positive": pos_text}
        for j, nid in enumerate(neg_ids[:n_neg]):
            row[f"negative_{j+1}"] = chunk_by_id[nid]["text"]
        rows.append(row)
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"    _build_training_rows: {i+1}/{n}  ({elapsed()/60:.1f} phút)")
    return rows


def enable_cuda_perf() -> None:
    """BẢN SỬA #5: thay cho `_cap_cuda_memory()` cũ (đặt trần 92% VRAM). Ở đây KHÔNG đặt
    trần gì — dùng thẳng 100% VRAM vật lý, chỉ bật các cờ tăng tốc THẬT (TF32 trên Tensor
    Core của Ampere, cudnn tự chọn thuật toán nhanh nhất cho shape cố định). An toàn dựa
    vào các vòng lặp OOM-backoff ở finetune_or_load_dense()/encode_corpus(): gặp
    CUDA OOM thật thì mới giảm batch và chạy lại, không đoán trước và giới hạn sẵn."""
    import torch
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [GPU] {torch.cuda.get_device_name(0)} — {total_gb:.1f}GB VRAM, dùng toàn bộ "
              f"(không đặt trần nhân tạo, chỉ lùi batch khi OOM thật xảy ra).")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # input shape cố định (đã pad theo DENSE_MAX_SEQ_LEN)
        # -> cuDNN tự chọn thuật toán nhanh nhất cho shape này sau vài batch đầu.


def finetune_or_load_dense(train_positive, train_data, chunk_by_id, all_chunks, bm25):
    import torch
    from sentence_transformers import SentenceTransformer

    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    if cuda_ok:
        print(f"  Device: cuda ({torch.cuda.get_device_name(0)})")
        enable_cuda_perf()
    else:
        print("  [CẢNH BÁO] torch.cuda.is_available()=False — không thấy GPU dù bạn có RTX 2050. "
              "Nguyên nhân thường gặp nhất: đã cài bản torch CPU-only. Sửa bằng lệnh "
              "(chạy trong venv 'env', gỡ torch cũ trước nếu có):\n"
              "    pip uninstall torch torchvision torchaudio -y\n"
              "    pip install torch --index-url https://download.pytorch.org/whl/cu130\n"
              "  (cu130 = CUDA 13.0, bản mặc định hiện tại — driver của bạn theo nvidia-smi hỗ trợ "
              "tới CUDA 13.3 nên cu130 chạy tốt; nếu vẫn lỗi, thử cu126 — driver mới luôn tương thích "
              "ngược với CUDA runtime cũ hơn). Đang tiếp tục chạy CPU — chậm hơn nhiều, time-boxing "
              "sẽ tự giảm số step.")

    use_finetune = len(train_positive) >= MIN_TRAIN_PAIRS and remaining() > 5 * 60
    if not use_finetune:
        print(f"  {len(train_positive)} positive pairs (< {MIN_TRAIN_PAIRS}) hoặc hết ngân sách "
              f"-> dùng zero-shot '{BASE_DENSE_MODEL}', không fine-tune.")
        model = SentenceTransformer(BASE_DENSE_MODEL, device=device)
        model.max_seq_length = DENSE_MAX_SEQ_LEN
        return model

    # `SentenceTransformerTrainer` (API chính thức bản 5.x) thay vì `.fit()` (API cũ, có thể
    # ImportError trên bản mới) — có `max_steps` cho time-boxing sạch.
    #
    # Bọc trong try/except: nếu thiếu package `datasets`, in rõ lệnh cần cài rồi TỰ ĐỘNG rơi
    # về zero-shot thay vì crash — vẫn có submission.zip ngay lần chạy này.
    model = SentenceTransformer(BASE_DENSE_MODEL, device=device)
    model.max_seq_length = DENSE_MAX_SEQ_LEN

    try:
        print("  Đang import datasets/Trainer/accelerate...")
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
        import accelerate  # noqa: F401 — cần >= 1.1.0 để dựng device setup (Trainer nền transformers).
        print("  Import xong.")
        if tuple(map(int, accelerate.__version__.split(".")[:2])) < (1, 1):
            raise ImportError(f"accelerate {accelerate.__version__} quá cũ, cần >= 1.1.0")
    except ImportError as e:
        print(f"  [THIẾU PACKAGE] {e}. Chạy: pip install datasets \"accelerate>=1.1.0\"\n"
              f"  -> Bỏ qua fine-tune LẦN NÀY, dùng zero-shot '{BASE_DENSE_MODEL}' để vẫn ra được "
              f"submission.zip. Cài xong package rồi chạy lại để có bản fine-tune tốt hơn.")
        return model

    if len(train_positive) > MAX_TRAIN_EXAMPLES:
        sampled_qids = random.sample(list(train_positive.keys()), MAX_TRAIN_EXAMPLES)
        train_positive_used = {qid: train_positive[qid] for qid in sampled_qids}
        print(f"  Có {len(train_positive)} positive pairs, lấy mẫu ngẫu nhiên {MAX_TRAIN_EXAMPLES} "
              f"để giới hạn thời gian mine hard-negative + train (xem MAX_TRAIN_EXAMPLES ở CONFIG).")
    else:
        train_positive_used = train_positive

    print(f"  Đang tạo training rows (BM25 top_k cho {len(train_positive_used)} câu hỏi)...")
    rows = _build_training_rows(train_positive_used, train_data, chunk_by_id, all_chunks, bm25)
    print("  Đang tạo Dataset object...")
    dataset = Dataset.from_list(rows)
    print(f"  Training rows: {len(dataset)}")

    batch_size = TRAIN_BATCH_SIZE          # batch HIỆU DỤNG — quyết định chất lượng train (số
                                            # in-batch negative), KHÔNG đổi khi gặp OOM.
    mini_batch_size = TRAIN_MINI_BATCH_SIZE  # batch THẬT xử lý mỗi lần forward — quyết định VRAM,
                                              # ĐÂY mới là thứ tự động giảm khi OOM.
    for attempt in range(4):  # tự giảm mini_batch_size nếu OOM (không đụng batch_size hiệu dụng)
        try:
            # CachedMultipleNegativesRankingLoss (GradCache): tách "batch hiệu dụng" (nhiều
            # in-batch negative, quyết định CHẤT LƯỢNG train) khỏi "mini-batch" (quyết định VRAM
            # THẬT dùng). Chạy chậm hơn ~20-30% (forward 2 lần) nhưng batch hiệu dụng lớn hơn
            # nhiều so với bị OOM ép xuống batch=1.
            loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)

            calib_steps = min(10, max(1, len(dataset) // batch_size))
            calib_args = SentenceTransformerTrainingArguments(
                output_dir=str(TRAINER_TMP_DIR), max_steps=calib_steps,
                per_device_train_batch_size=batch_size, logging_steps=calib_steps + 1,
                save_strategy="no", report_to=[], disable_tqdm=True,
                fp16=(device == "cuda"),
            )
            calib_start = time.time()
            print(f"  Đang chạy calib training (batch hiệu dụng={batch_size}, mini_batch={mini_batch_size}, "
                  f"lần đầu init CUDA context có thể mất 10-30s, sau đó nhanh)...")
            SentenceTransformerTrainer(model=model, args=calib_args, train_dataset=dataset, loss=loss).train()
            calib_time = (time.time() - calib_start) / calib_steps

            budget_left = min(remaining() - 3 * 60, FINETUNE_TIME_BUDGET_SEC - (time.time() - calib_start))
            max_steps = max(0, int(budget_left / max(calib_time, 1e-6)))
            max_steps = min(max_steps, (len(dataset) // batch_size) * 8)  # cap 8 epoch — chỉ để
            # tránh lặp dữ liệu vô hạn nếu máy nhanh bất thường, time-boxing thật ở trên vẫn là
            # ràng buộc chính khi máy chậm hơn.
            print(f"  Calib: ~{calib_time:.2f}s/step ({device}), ngân sách còn ~{budget_left/60:.1f} phút "
                  f"-> chạy thêm tối đa {max_steps} step (batch hiệu dụng={batch_size}).")

            if max_steps > 0:
                args = SentenceTransformerTrainingArguments(
                    output_dir=str(TRAINER_TMP_DIR), max_steps=max_steps,
                    per_device_train_batch_size=batch_size, learning_rate=2e-5,
                    warmup_steps=0.05,  # float = tỉ lệ warmup (API mới, tránh deprecation warning)
                    lr_scheduler_type="cosine",
                    logging_steps=max(1, max_steps // 20), save_strategy="no", report_to=[],
                    fp16=(device == "cuda"),
                )
                SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss).train()
            break
        except (RuntimeError, ImportError) as e:
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower() and mini_batch_size > 1:
                print(f"  [CUDA OOM] mini_batch_size={mini_batch_size} quá lớn -> thử "
                      f"mini_batch_size={mini_batch_size // 2} (batch hiệu dụng {batch_size} GIỮ NGUYÊN — "
                      f"chỉ mini-batch xử lý thật mới giảm, nhờ CachedMultipleNegativesRankingLoss).")
                if device == "cuda":
                    torch.cuda.empty_cache()
                mini_batch_size = max(1, mini_batch_size // 2)
                continue
            if isinstance(e, ImportError):
                print(f"  [THIẾU PACKAGE lúc train] {e} -> dùng zero-shot thay vì crash.")
                return model
            raise

    # BẢN SỬA #5: KHÔNG lưu checkpoint ra đĩa (`model.save_pretrained(...)` đã bỏ) — model
    # fine-tune xong dùng NGAY cho Bước 5 (encode corpus) trong cùng lần chạy, không cần
    # persist. Vẫn giữ việc ép lại device: bản #4 phát hiện `save_pretrained()` có thể âm
    # thầm để model kẹt ở CPU sau khi lưu (safetensors cần CPU liền mạch để serialize) —
    # dù giờ không save nữa, ép + xác nhận device vẫn là lưới an toàn rẻ, giữ nguyên.
    model = model.to(device)
    actual_device = next(model.parameters()).device
    print(f"  [Xác nhận device sau train] model đang ở: {actual_device} (kỳ vọng: {device})")
    # So .type (chỉ lấy "cuda"/"cpu", bỏ index) — so thẳng chuỗi "cuda:0" != "cuda" là false
    # positive đã gặp thật.
    if actual_device.type != device and device == "cuda":
        print("  [CẢNH BÁO NGHIÊM TRỌNG] model vẫn KHÔNG ở GPU sau khi ép .to(device) — "
              "kiểm tra lại cài đặt torch/CUDA, Bước 5 sẽ CHẬM nếu tiếp tục ở CPU.")
    return model


# ==============================================================================
# BƯỚC 5 — Encode toàn bộ corpus + RRF fusion (BM25 + dense), KHÔNG rerank (xem lý do ở đầu file)
# ==============================================================================
def encode_corpus(model, all_chunks: list):
    import torch
    if torch.cuda.is_available():
        enable_cuda_perf()  # BẢN SỬA #5: chỉ bật cờ tăng tốc, KHÔNG đặt trần VRAM nữa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        model = model.half()  # fp16 cho encode — giảm ~1 nửa VRAM cần dùng + nhanh hơn nhờ
        # Tensor Core (Ampere), cho phép batch lớn hơn. Không dùng cho CPU (fp16 trên CPU
        # thường CHẬM hơn fp32, không có tăng tốc phần cứng tương ứng).
    actual_device = next(model.parameters()).device
    print(f"  [Xác nhận device trước khi encode] model đang ở: {actual_device}"
          f"{' (fp16)' if device == 'cuda' else ''}")
    if actual_device.type != device and device == "cuda":
        raise RuntimeError(
            f"model vẫn ở {actual_device} thay vì cuda sau khi .to('cuda') — dừng lại thay vì "
            f"âm thầm chạy CPU nhiều giờ. Kiểm tra lại cài đặt torch (torch.cuda.is_available() "
            f"phải True) trước khi chạy lại."
        )

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
                print(f"  [CUDA OOM] encode batch_size={batch_size} -> thử {batch_size // 2}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            raise


def load_reranker():
    """Dùng ZERO-SHOT `AITeamVN/Vietnamese_Reranker` (fine-tune từ bge-reranker-v2-m3, huấn
    luyện trên "toàn bộ tập Legal Zalo 2021" — cùng domain pháp lý tiếng Việt với đề thi này)
    thay vì tự fine-tune — model đã chuyên biệt sẵn cho đúng domain, tránh thêm 1 vòng train
    tốn thời gian + rủi ro VRAM (model nền tảng bge-reranker-v2-m3 lớn hơn nhiều, ~568M, so
    với ~135M của retriever).
    Trả về (None, None) nếu tải lỗi (mạng/OOM) — answer_question() sẽ tự bỏ qua rerank, không
    crash toàn bộ pipeline vì 1 model tuỳ chọn."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    for attempt in range(2):  # thử lại 1 lần nếu lỗi mạng thoáng qua
        try:
            print(f"  Đang tải reranker AITeamVN/Vietnamese_Reranker (zero-shot, không fine-tune)"
                  f"{' — thử lại lần 2' if attempt else ''}...")
            tokenizer = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Reranker")
            model = AutoModelForSequenceClassification.from_pretrained("AITeamVN/Vietnamese_Reranker")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            if device == "cuda":
                model = model.half()
            model.eval()
            print(f"  Reranker sẵn sàng trên {next(model.parameters()).device}"
                  f"{' (fp16)' if device == 'cuda' else ''}.")
            return model, tokenizer
        except Exception as e:
            if attempt == 0:
                print(f"  [Lần 1 lỗi: {e}] thử lại sau 5s...")
                time.sleep(5)
                continue
            print(f"  [CẢNH BÁO] Không tải được reranker sau 2 lần thử ({e}) -> bỏ qua rerank, "
                  f"dùng thẳng thứ hạng RRF (BM25+dense). Không ảnh hưởng tới việc ra submission.zip. "
                  f"Nếu lỗi vẫn là 'couldn't connect', kiểm tra Internet thật sự đang bật.")
            return None, None


def rerank(question: str, candidates: list, reranker_model, reranker_tokenizer,
           max_candidates: int = 100, max_length: int = 1024):
    """Chấm điểm lại top `max_candidates` bằng cross-encoder, trả về (list đã sắp xếp lại,
    mảng điểm số tương ứng — None nếu không rerank được). max_candidates=100 (=TOP_K_RETRIEVE):
    đo được Recall@30=78.3% nhưng Recall@100=85.0% — giới hạn 30 tự bỏ lỡ 6.7 điểm % chunk
    đúng nằm ở rank 31-100. max_length=1024 (không dùng hết 2304 model hỗ trợ) — cân bằng
    tốc độ/VRAM.
    Trả thêm điểm số để adaptive_k_cutoff() có thể tìm điểm "gãy" tự nhiên trong phân phối
    điểm, thay vì luôn dùng top_n cố định."""
    import torch
    if reranker_model is None or not candidates:
        return candidates, None
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
        sorted_scores = scores[order]
        return reranked + candidates[max_candidates:], sorted_scores
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  [CUDA OOM lúc rerank] bỏ qua rerank cho câu hỏi này, dùng thứ hạng RRF.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return candidates, None
        raise


def adaptive_k_cutoff(scores, min_k: int = 1, max_k: int = 5, search_window: int = 15) -> int:
    """Adaptive-k (Taguchi et al. 2025, arXiv:2506.08479): thay vì top_n CỐ ĐỊNH cho mọi câu
    hỏi, tìm điểm "gãy" tự nhiên (largest gap) trong phân phối điểm reranker đã sort giảm dần:
        k* = argmax_i (score[i] - score[i+1])
    Trực giác: nếu chunk đúng có điểm cao rõ rệt rồi tới khoảng trống lớn trước khi rớt xuống
    điểm thấp (chunk không liên quan), điểm gãy đó là ranh giới tự nhiên. Đo được: top_n cố
    định làm METEOR đạt đỉnh ở top_n=2-3 rồi giảm mạnh — với câu hỏi mà điểm gãy thật nằm ở
    vị trí 1, ép thêm chunk 2-3 (không liên quan) làm precision sập.
    Giới hạn [min_k, max_k]: tránh k=0 (không hợp lệ) và k quá lớn khi phân phối điểm gần
    phẳng (không có gap rõ ràng -> an toàn hơn là dùng ít chunk)."""
    if scores is None or len(scores) == 0:
        return min_k
    n = min(len(scores), search_window)
    if n <= 1:
        return min_k
    gaps = [scores[i] - scores[i + 1] for i in range(n - 1)]
    k_star = int(np.argmax(gaps)) + 1  # +1: index 0 nghĩa là gap SAU phần tử thứ 1 -> k=1
    return max(min_k, min(k_star, max_k))


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
# BƯỚC 6 — Sinh câu trả lời (template extractive, dùng so_hieu/loai_vb thật — KHÔNG dùng tên file)
# ==============================================================================
_DIEU_PREFIX_STRIP_RE = re.compile(r"^\s*Điều\s+\d+[a-zđA-ZĐ]?\.?\s*", re.IGNORECASE)


def render_answer(selected_chunks: list, top_n: int) -> str:
    """Đo tần suất cụm mở đầu + kết câu dẫn trên toàn bộ 7000 answer thật:
      - Mở đầu bằng "Căn cứ": 57.4% (4020/7000)  vs  "Theo": 25.1% (1759/7000)
      - Cụm "quy định như sau": 24.6% (1720/7000)  vs  "quy định cụ thể": chỉ 1.1% (75/7000)
    Dùng khuôn phổ biến nhất "Căn cứ Điều X <loại VB> <số hiệu> quy định như sau:".

    Dòng NGAY SAU câu dẫn: 98.8% (5954/6026 answer có pattern này) là TIÊU ĐỀ TRẦN của Điều,
    KHÔNG lặp lại "Điều X.". Corpus chunk luôn bắt đầu bằng "Điều X. <tiêu đề>" — cắt bỏ phần
    "Điều X." ở ĐẦU text hiển thị (KHÔNG đụng text_raw gốc dùng cho việc khác) để khớp đúng
    định dạng thật."""
    parts, seen = [], set()
    for c in selected_chunks:
        if c["id"] in seen or len(parts) >= top_n:
            continue
        seen.add(c["id"])
        loai_vb = c["loai_vb"] or "văn bản"
        so_hieu = c["so_hieu"] or ""
        dieu = c["dieu_so"]
        lead = (f"Căn cứ Điều {dieu} {loai_vb} {so_hieu} quy định như sau:"
                if dieu != "0" else f"Căn cứ {loai_vb} {so_hieu} quy định như sau:")
        body = _DIEU_PREFIX_STRIP_RE.sub("", c["text"], count=1) if dieu != "0" else c["text"]
        parts.append(f"{lead}\n{body}")
    return "\n\n".join(parts)


def answer_question(question: str, bm25, dense_model, dense_embeddings, all_chunks, top_n: int,
                     reranker_model=None, reranker_tokenizer=None, use_adaptive_k: bool = False) -> str:
    ranked = rrf_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks)
    if not ranked:
        return "Không tìm thấy thông tin pháp lý cho câu hỏi này."
    scores = None
    if reranker_model is not None:
        ranked, scores = rerank(question, ranked, reranker_model, reranker_tokenizer)
    n = adaptive_k_cutoff(scores) if (use_adaptive_k and scores is not None) else top_n
    return render_answer(ranked, n)


# ==============================================================================
# BƯỚC 7 — Dev-eval (METEOR/ROUGE-L) trên mẫu train.json để chọn TOP_N_ANSWER, và
# BƯỚC 8 — Validate + đóng gói submission.zip
# ==============================================================================
def measure_retrieval_recall(bm25, dense_model, dense_embeddings, all_chunks, train_data,
                              train_positive, reranker_model=None, reranker_tokenizer=None,
                              sample_size: int = 300) -> None:
    """Đo TRỰC TIẾP retrieval có tìm đúng chunk hay không, tách biệt khỏi METEOR (vốn trộn
    lẫn cả lỗi retrieval LẪN lỗi câu chữ). Dùng chính `train_positive` (chunk_id đúng, suy từ
    citation trong answer thật) làm ground truth — đo Recall@k tại nhiều mức k, CÓ và KHÔNG
    rerank, để biết chắc nút thắt nằm ở retriever hay ở thứ hạng cuối."""
    ids = [qid for qid in random.sample(list(train_positive.keys()),
                                          min(sample_size, len(train_positive)))]
    print(f"  Đo Recall@k trên {len(ids)} câu hỏi có nhãn đúng (từ citation trong train.json)...")

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
            reranked, _scores = rerank(question, ranked, reranker_model, reranker_tokenizer)
            reranked_ids = [c["id"] for c in reranked]
            for k in ks:
                if pos_id in reranked_ids[:k]:
                    hits_rerank[k] += 1

    n = len(ids)
    print("  --- Recall@k: BM25+dense (RRF, chưa rerank) ---")
    for k in ks:
        print(f"    Recall@{k:<3d} = {hits_rrf[k]}/{n} = {100*hits_rrf[k]/n:.1f}%")
    if hits_rerank is not None:
        print("  --- Recall@k: sau rerank ---")
        for k in ks:
            print(f"    Recall@{k:<3d} = {hits_rerank[k]}/{n} = {100*hits_rerank[k]/n:.1f}%")
    print("  Diễn giải: Recall@100 thấp -> vấn đề ở retriever (BM25/dense không tìm thấy chunk "
          "đúng trong 100 ứng viên đầu) -> cần cải thiện retriever, KHÔNG phải template/rerank. "
          "Recall@100 cao nhưng Recall@5 thấp -> retriever tìm được nhưng xếp hạng chưa tốt -> "
          "rerank/threshold là chỗ cần cải thiện.")


def try_dev_eval(bm25, dense_model, dense_embeddings, all_chunks, train_data,
                  reranker_model=None, reranker_tokenizer=None) -> tuple:
    try:
        import nltk
        if str(NLTK_CACHE_DIR) not in nltk.data.path:
            nltk.data.path.insert(0, str(NLTK_CACHE_DIR))
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True, download_dir=str(NLTK_CACHE_DIR))
            nltk.download("omw-1.4", quiet=True, download_dir=str(NLTK_CACHE_DIR))
        from nltk.translate.meteor_score import meteor_score
        from rouge_score import rouge_scorer
    except Exception as e:
        print(f"  Bỏ qua dev-eval (thiếu nltk/rouge_score: {e}). Dùng TOP_N_ANSWER=3, không rerank.")
        return 3, False

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    random.seed(42)
    n_sample = min(DEV_EVAL_SAMPLE_SIZE, len(train_data))
    ids = random.sample(list(train_data.keys()), n_sample)

    # retrieval + rerank (bước ĐẮT nhất) chỉ chạy 1 LẦN/câu hỏi/config, rồi thử nhiều top_n
    # bằng cách CẮT list đã xếp hạng (render_answer rất rẻ).
    configs = [("BM25+dense (không rerank)", None, None)]
    if reranker_model is not None:
        configs.append(("BM25+dense+rerank", reranker_model, reranker_tokenizer))

    best_n, best_m, best_use_rerank, best_use_adaptive = 3, -1.0, False, False
    for label, rr_model, rr_tok in configs:
        print(f"  --- {label} ---")
        ranked_cache, scores_cache = {}, {}
        for qid in ids:
            item = train_data[qid]
            ranked = rrf_retrieve(item["question"], bm25, dense_model, dense_embeddings, all_chunks)
            scores = None
            if rr_model is not None and ranked:
                ranked, scores = rerank(item["question"], ranked, rr_model, rr_tok)
            ranked_cache[qid] = ranked
            scores_cache[qid] = scores

        for top_n in (1, 2, 3, 4, 5):  # đỉnh METEOR quan sát được quanh top_n=3
            ms, rs = [], []
            for qid in ids:
                ranked = ranked_cache[qid]
                pred = render_answer(ranked, top_n) if ranked else "Không tìm thấy thông tin pháp lý cho câu hỏi này."
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms) / len(ms), sum(rs) / len(rs)
            print(f"    top_n={top_n}  METEOR={m:.4f}  ROUGE-L={r:.4f}  (n={len(ids)})")
            if m > best_m:
                best_m, best_n, best_use_rerank, best_use_adaptive = m, top_n, (rr_model is not None), False

        if rr_model is not None:
            ms, rs = [], []
            for qid in ids:
                ranked, scores = ranked_cache[qid], scores_cache[qid]
                k = adaptive_k_cutoff(scores) if ranked else 1
                pred = render_answer(ranked, k) if ranked else "Không tìm thấy thông tin pháp lý cho câu hỏi này."
                ref = train_data[qid]["answer"]
                ms.append(meteor_score([str(ref).split()], str(pred).split()))
                rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
            m, r = sum(ms) / len(ms), sum(rs) / len(rs)
            print(f"    adaptive-k       METEOR={m:.4f}  ROUGE-L={r:.4f}  (n={len(ids)})")
            if m > best_m:
                best_m, best_use_rerank, best_use_adaptive = m, True, True

    print(f"  => chọn TOP_N_ANSWER={best_n}, dùng reranker={best_use_rerank}, "
          f"dùng adaptive-k={best_use_adaptive} (METEOR={best_m:.4f})")
    return best_n, best_use_rerank, best_use_adaptive


def build_submission(answers: dict, expected_ids: set, out_zip: Path) -> None:
    errors = []
    got = set(answers.keys())
    if got != expected_ids:
        errors.append(f"Key lệch: thiếu {len(expected_ids-got)}, thừa {len(got-expected_ids)}")
    for qid, ans in answers.items():
        if not isinstance(ans, str) or not ans.strip():
            errors.append(f"[{qid}] answer rỗng hoặc không phải string")
    if errors:
        raise ValueError("Submission KHÔNG hợp lệ:\n  - " + "\n  - ".join(errors[:20]))

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
    print(f"  OK — {out_zip} ({len(normalized)} câu trả lời, đã kiểm tra lại từ đĩa)")


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    checkpoint("Bắt đầu")
    print(f"  Cache/model tải về: {CACHE_DIR}")

    print("\n=== Bước 1: Chunk corpus ===")
    all_chunks = load_corpus(CONTEXTS_DIR)
    checkpoint("Xong chunking")

    print("\n=== Bước 2: BM25 index ===")
    tokenized = [tokenize_simple(f"{c.get('loai_vb','')} {c['text']}") for c in all_chunks]
    bm25 = BM25(tokenized)
    checkpoint("Xong BM25 index")

    print("\n=== Bước 3: Sinh nhãn từ train.json ===")
    with TRAIN_PATH.open(encoding="utf-8") as f:
        train_data = json.load(f)
    train_positive, chunk_by_id = build_train_pairs(train_data, all_chunks)
    print(f"  Positive pairs: {len(train_positive)}/{len(train_data)}")
    checkpoint("Xong sinh nhãn")

    print("\n=== Bước 4: Fine-tune (hoặc load zero-shot) dense retriever ===")
    dense_model = finetune_or_load_dense(train_positive, train_data, chunk_by_id, all_chunks, bm25)
    checkpoint("Xong Bước 4 (dense retriever)")

    print("\n=== Bước 5: Encode toàn bộ corpus ===")
    dense_embeddings = encode_corpus(dense_model, all_chunks)
    checkpoint("Xong encode corpus")

    print("\n=== Bước 5b: Tải reranker (zero-shot, không fine-tune) ===")
    reranker_model, reranker_tokenizer = load_reranker()
    checkpoint("Xong tải reranker")

    print("\n=== Bước 5c: Đo Recall@k — retrieval có tìm đúng chunk không? ===")
    measure_retrieval_recall(bm25, dense_model, dense_embeddings, all_chunks, train_data,
                              train_positive, reranker_model, reranker_tokenizer)
    checkpoint("Xong đo Recall@k")

    print("\n=== Bước 6: Dev-eval chọn TOP_N_ANSWER + xác nhận reranker có giúp ích không ===")
    top_n_answer, use_reranker, use_adaptive = try_dev_eval(bm25, dense_model, dense_embeddings, all_chunks,
                                                             train_data, reranker_model, reranker_tokenizer)
    checkpoint("Xong dev-eval")

    print("\n=== Bước 7: Sinh câu trả lời cho public-official.json ===")
    with PUBLIC_PATH.open(encoding="utf-8") as f:
        questions = json.load(f)
    rr_model = reranker_model if use_reranker else None
    rr_tok = reranker_tokenizer if use_reranker else None
    answers = {}
    for i, (qid, item) in enumerate(questions.items()):
        answers[qid] = answer_question(item["question"], bm25, dense_model, dense_embeddings,
                                        all_chunks, top_n_answer, reranker_model=rr_model,
                                        reranker_tokenizer=rr_tok, use_adaptive_k=use_adaptive)
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(questions)}  ({elapsed()/60:.1f} phút)")
    n_empty = sum(1 for a in answers.values() if not a.strip())
    print(f"  Đã sinh {len(answers)} câu trả lời, {n_empty} câu rỗng")
    checkpoint("Xong sinh câu trả lời")

    print("\n=== Bước 8: Đóng gói submission.zip ===")
    build_submission(answers, set(questions.keys()), OUT_DIR / "submission.zip")
    checkpoint(f"XONG — tổng thời gian {elapsed()/60:.1f} phút (trần an toàn {TIME_BUDGET_SEC/3600:.0f} giờ)")


if __name__ == "__main__":
    main()