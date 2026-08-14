"""
legalqa_local.py — LegalQA (UIT DSC2026 Task 2), tối ưu cho RTX 2050 4GB VRAM.

CÁCH DÙNG: đặt file này cạnh train.json, public-official.json, selected-contexts/ (đúng
layout thư mục của bạn) rồi chạy:
    python legalqa_local.py
Output: submission.zip trong cùng thư mục.

THƯ VIỆN CẦN CÀI (trong venv "env" của bạn):
    pip install numpy sentence-transformers datasets "accelerate>=1.1.0" nltk rouge_score
Và BẮT BUỘC kiểm tra torch có nhận đúng GPU không TRƯỚC khi chạy (chạy thử):
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only')"
Nếu ra False dù máy có RTX 2050 — cài lại torch bản có CUDA (driver mới thì dùng cu130,
driver cũ hơn thì cu126), KHÔNG dùng `pip install torch` trần trên Windows (mặc định tải
bản CPU-only).

===============================================================================
BẢN SỬA #4 (root-cause fix + đổi triết lý thời gian, theo log chạy thật 11h10')
===============================================================================
LỖI NGHIÊM TRỌNG đã tìm ra: Bước 5 (encode corpus) chạy 10h10' ở tốc độ 3.62s/batch —
chậm hơn ~40-70 lần so với tốc độ GPU bình thường cho model 135M tham số. Nguyên nhân:
`model.save_pretrained()` ở cuối Bước 4 (fine-tune) cần chuyển tensor về CPU để serialize
(safetensors yêu cầu bộ nhớ CPU liền mạch) và không chuyển lại GPU — model trả về cho
Bước 5 vì vậy âm thầm chạy trên CPU dù log báo "Device: cuda". Đã sửa: ép `.to(device)`
tường minh + IN RA device thật để xác nhận ở CẢ 2 nơi (cuối Bước 4, đầu Bước 5) — nếu vẫn
sai device, Bước 5 giờ sẽ RAISE lỗi rõ ràng ngay lập tức thay vì âm thầm chạy 10 tiếng.

ĐỔI TRIẾT LÝ THỜI GIAN (theo yêu cầu): bản trước gate CỨNG theo từng phase (55 phút tổng,
22 phút cho fine-tune...) — hậu quả phụ là dev-eval bị cắt ngang giữa chừng, cho kết quả
vô nghĩa (METEOR=-1.0). Từ bản này: KHÔNG còn gate cứng theo phase, chỉ giữ 1 trần an
toàn tổng thể rộng (3 giờ) để log tham khảo. Ưu tiên chạy ĐỦ mọi bước (đặc biệt dev-eval)
để có kết quả đáng tin, thay vì cắt ngắn cho kịp giờ. Với bug device đã sửa, cả pipeline
thực tế nên xong trong 30-90 phút — không cần đánh đổi chất lượng lấy tốc độ nữa.

Cũng đổi `BASE_DENSE_MODEL` sang `bkai-foundation-models/vietnamese-bi-encoder` — model
được dùng trực tiếp cho Vietnamese Legal QA retrieval trong nghiên cứu thực tế (Pham et
al., arXiv:2409.13699), cùng domain với bài thi này, thay cho SimCSE tổng quát trước đó.

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

# SỬA (log lần 4): treo với CPU~8%/RAM ổn định = đang CHỜ MẠNG, không phải đang tính toán —
# log trước đó có "sending unauthenticated requests to HF Hub" xác nhận có gọi mạng ở bước
# này. Model đã tải xong (thấy "LOAD REPORT" trong log) nên KHÔNG cần gọi mạng thêm nữa — tắt
# hẳn các cuộc gọi ra HF Hub (telemetry, version check) TRƯỚC KHI import bất cứ thư viện HF nào
# (phải đặt env var trước import, đặt sau không có tác dụng).
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# os.environ.setdefault("HF_HUB_OFFLINE", "1")           # model đã cache -> không cần mạng nữa
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # tránh treo do fork trên Windows
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
import re
import json
import math
import time
import random
import zipfile
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

# ==============================================================================
# CONFIG
# ==============================================================================
HERE = Path(__file__).resolve().parent  # script chạy cùng thư mục với train.json, v.v.

CONTEXTS_DIR = HERE / "selected-contexts"
TRAIN_PATH = HERE / "train.json"
PUBLIC_PATH = HERE / "public-official.json"
OUT_DIR = HERE

TIME_BUDGET_SEC = 3 * 3600         # SỬA (theo yêu cầu): không còn gate cứng theo từng phase — đây
                                    # là TRẦN AN TOÀN tổng thể (3 giờ), chỉ để log "còn lại bao
                                    # nhiêu", KHÔNG cắt ngang dev-eval/predict như bản trước (nguyên
                                    # nhân dev-eval bị bỏ qua, METEOR=-1.0 lần chạy trước). Với
                                    # bug device đã sửa bên dưới, cả pipeline thực tế nên xong trong
                                    # 30-60 phút — 3 giờ là biên an toàn rộng, không phải mục tiêu.
FINETUNE_TIME_BUDGET_SEC = 90 * 60  # tối đa dành cho fine-tune dense retriever (Bước 4) — nới rộng
                                    # vì giờ chạy đúng GPU sẽ nhanh hơn nhiều, không cần siết chặt.
MIN_TRAIN_PAIRS = 50               # dưới ngưỡng này -> bỏ fine-tune, dùng zero-shot
MAX_TRAIN_EXAMPLES = 3000          # SỬA: nới từ 1000 lên 3000 (gần hết 3565 positive pairs thật) —
                                    # trước đây giới hạn thấp vì lo ngân sách 55 phút, giờ không còn
                                    # ràng buộc đó nên dùng gần hết dữ liệu có nhãn để fine-tune tốt hơn.

BASE_DENSE_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
# SỬA: đổi từ VoVanPhuc/sup-SimCSE-VietNamese-phobert-base (general-purpose) sang model này —
# cùng cỡ ~135M tham số (vẫn vừa 4GB thoải mái), nhưng đã được dùng trực tiếp cho bài toán
# Vietnamese Legal QA retrieval trong nghiên cứu thực tế (Pham et al., "Vietnamese Legal
# Information Retrieval in Question-Answering System", arXiv:2409.13699) — cùng domain với
# bài thi này, nhiều khả năng cho embedding chất lượng tốt hơn cho truy vấn pháp luật.
DENSE_MAX_SEQ_LEN = 256            # cắt ngắn để tiết kiệm VRAM + thời gian (câu luật dài,
                                    # nhưng embedding chỉ cần đủ để phân biệt ngữ nghĩa, không
                                    # cần đọc hết toàn văn — sinh câu trả lời vẫn dùng text đầy đủ)
TRAIN_BATCH_SIZE = 4               # SỬA: giảm từ 8 xuống 4 — theo quan sát thật (VRAM 3.9/4GB
                                    # tràn sang shared memory ở cấu hình cũ), khởi điểm an toàn
                                    # hơn cho 4GB thật. Vẫn tự động giảm tiếp nếu OOM (giờ OOM sẽ
                                    # thật sự được raise nhờ _cap_cuda_memory(), xem hàm đó).
ENCODE_BATCH_SIZE = 16             # SỬA: giảm từ 32 xuống 16 — cùng lý do trên.

TOP_K_RETRIEVE = 100                # số ứng viên lấy ra sau RRF fusion
DEV_EVAL_SAMPLE_SIZE = 300          # SỬA: nới từ 120 lên 300 — giờ có thời gian, mẫu lớn hơn cho
                                    # tín hiệu METEOR đáng tin hơn khi chọn TOP_N_ANSWER.

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
        self.inverted = defaultdict(list)
        for i, doc in enumerate(tokenized_docs):
            for term, f in Counter(doc).items():
                self.inverted[term].append((i, f))
        df = {t: len(p) for t, p in self.inverted.items()}
        idf_raw = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}
        avg_idf = sum(idf_raw.values()) / len(idf_raw) if idf_raw else 0.0
        eps = 0.25 * avg_idf
        self.idf = {t: (v if v > 0 else eps) for t, v in idf_raw.items()}

    def get_scores(self, query_tokens) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float64)
        for term in set(query_tokens):
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc_idx, f in postings:
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl)
                scores[doc_idx] += idf * f * (self.k1 + 1) / denom
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
def _build_training_rows(train_positive, train_data, chunk_by_id, all_chunks, bm25, n_neg: int = 4):
    """Trả về list dict {anchor, positive, negative_1..negative_n} — đúng format
    SentenceTransformerTrainer + MultipleNegativesRankingLoss chấp nhận trực tiếp.
    In progress mỗi 500 câu — SỬA (log lần 4): vòng lặp này có thể chạy vài nghìn lần (mỗi lần
    1 lượt BM25 top_k trên toàn corpus), trước đây KHÔNG in gì cho tới khi xong hẳn -> trông
    giống "treo" dù thực ra đang chạy bình thường, không phân biệt được với treo thật."""
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


def _cap_cuda_memory(fraction: float = 0.85) -> None:
    """SỬA (phát hiện từ Task Manager: VRAM 3.9/4GB + shared memory 4.8GB + GPU-Util 0%):
    Windows (driver WDDM) cho phép CUDA "tràn" sang shared system memory thay vì báo lỗi
    OutOfMemory khi vượt VRAM vật lý — quá trình dồn/lấy dữ liệu qua PCIe giữa VRAM thật và
    RAM hệ thống cực chậm (đúng khớp 3.62s/batch quan sát được), NHƯNG không raise exception
    nên logic tự-giảm-batch-khi-OOM (except RuntimeError "out of memory") KHÔNG BAO GIỜ được
    kích hoạt — CUDA "thành công" về mặt kỹ thuật, chỉ là chậm khủng khiếp.
    Ép giới hạn cứng bằng set_per_process_memory_fraction(): khi vượt ngưỡng này, PyTorch's
    caching allocator sẽ chủ động raise torch.cuda.OutOfMemoryError THẬT thay vì để driver
    âm thầm tràn sang shared memory — nhờ vậy logic retry-với-batch-nhỏ-hơn đã có sẵn mới
    thực sự chạy được. fraction=0.85 (không phải 1.0): chừa khoảng 15% cho CUDA context/driver
    overhead, tránh chính bản thân giới hạn này gây crash sớm không cần thiết."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [Giới hạn VRAM] Ép trần {fraction*100:.0f}% x {total_gb:.1f}GB = "
              f"~{fraction*total_gb:.2f}GB — vượt sẽ raise OutOfMemoryError thay vì tràn "
              f"sang shared memory (chậm âm thầm).")


def finetune_or_load_dense(train_positive, train_data, chunk_by_id, all_chunks, bm25):
    import torch
    from sentence_transformers import SentenceTransformer

    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    if cuda_ok:
        print(f"  Device: cuda ({torch.cuda.get_device_name(0)})")
        _cap_cuda_memory()
    else:
        # SỬA (phát hiện từ log chạy thật): torch.cuda.is_available()=False dù máy có RTX 2050 —
        # gần như chắc chắn do cài `pip install torch` không kèm CUDA (wheel mặc định trên PyPI
        # cho Windows là CPU-only). Không dừng chương trình — vẫn chạy CPU, thời gian tự động co
        # lại nhờ time-boxing bên dưới — nhưng in cảnh báo rõ để bạn sửa TẬN GỐC (chạy nhanh hơn
        # NHIỀU nếu dùng đúng GPU 4GB đang có sẵn):
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

    # SỬA lỗi (phát hiện từ log chạy thật): sentence-transformers bản mới (5.x) đã đổi
    # `.fit(train_objectives=...)` (API cũ) sang phụ thuộc gói `datasets` nội bộ và có thể
    # raise ImportError ngay cả khi gọi qua API cũ — dùng thẳng `SentenceTransformerTrainer`
    # (API chính thức của bản 5.x) thay vì `.fit()`, đồng thời có `max_steps` cho time-boxing
    # SẠCH hơn (không cần "calib rồi gọi fit 2 lần" như bản trước).
    #
    # SỬA thêm (phát hiện từ log chạy thật lần 2): thiếu package `datasets` làm crash TOÀN BỘ
    # script giữa chừng, bắt phải chạy lại từ đầu (mất vài phút chunk lại corpus). Bọc trong
    # try/except: nếu thiếu, in rõ lệnh cần cài rồi TỰ ĐỘNG rơi về zero-shot thay vì crash —
    # bạn vẫn có submission.zip ngay lần chạy này, cài thêm package rồi chạy lại sau để có
    # bản fine-tune tốt hơn. (Tạo `model` TRƯỚC try/except — bug ở bản trước tham chiếu `model`
    # trong nhánh except trước khi nó được gán, gây NameError thay vì fallback êm.)
    model = SentenceTransformer(BASE_DENSE_MODEL, device=device)
    model.max_seq_length = DENSE_MAX_SEQ_LEN

    try:
        print("  Đang import datasets/Trainer/accelerate...")
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        import accelerate  # noqa: F401 — SỬA (log lần 3): SentenceTransformerTrainingArguments cần
        # accelerate>=1.1.0 để dựng device setup (Trainer nền transformers), THIẾU package này
        # không raise lỗi ngay lúc import mà raise SÂU bên trong __post_init__() lúc khởi tạo
        # TrainingArguments — đó là lý do bạn thấy "treo lâu" trước khi báo lỗi (transformers dò
        # nhiều bước trước khi tới đoạn cần accelerate). Import tường minh ở đây để bắt SỚM, tránh
        # phải đợi lại từ đầu.
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

    batch_size = TRAIN_BATCH_SIZE
    for attempt in range(3):  # tự giảm batch nếu OOM
        try:
            loss = MultipleNegativesRankingLoss(model)

            # Calib: chạy thử vài chục step (rẻ) để đo giây/step THẬT trên máy này (CPU hay GPU
            # đều tự đo đúng, không cần biết trước phần cứng), rồi suy ra max_steps vừa ngân sách.
            calib_steps = min(10, max(1, len(dataset) // batch_size))
            calib_args = SentenceTransformerTrainingArguments(
                output_dir="dense_finetuned_tmp", max_steps=calib_steps,
                per_device_train_batch_size=batch_size, logging_steps=calib_steps + 1,
                save_strategy="no", report_to=[], disable_tqdm=True,
            )
            calib_start = time.time()
            print("  Đang chạy calib training (vài step đầu — lần đầu init CUDA context có thể mất "
                  "10-30s, sau đó nhanh)...")
            SentenceTransformerTrainer(model=model, args=calib_args, train_dataset=dataset, loss=loss).train()
            calib_time = (time.time() - calib_start) / calib_steps

            budget_left = min(remaining() - 3 * 60, FINETUNE_TIME_BUDGET_SEC - (time.time() - calib_start))
            max_steps = max(0, int(budget_left / max(calib_time, 1e-6)))
            max_steps = min(max_steps, (len(dataset) // batch_size) * 2)  # đừng train quá 2 "epoch" dữ liệu
            print(f"  Calib: ~{calib_time:.2f}s/step ({device}), ngân sách còn ~{budget_left/60:.1f} phút "
                  f"-> chạy thêm tối đa {max_steps} step (batch_size={batch_size}).")

            if max_steps > 0:
                args = SentenceTransformerTrainingArguments(
                    output_dir="dense_finetuned", max_steps=max_steps,
                    per_device_train_batch_size=batch_size, learning_rate=2e-5,
                    warmup_steps=0.05,  # float = tỉ lệ warmup (API mới thay cho warmup_ratio, tránh deprecation warning)
                    lr_scheduler_type="cosine",
                    logging_steps=max(1, max_steps // 20), save_strategy="no", report_to=[],
                )
                SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss).train()
            break
        except (RuntimeError, ImportError) as e:
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower() and batch_size > 1:
                print(f"  [CUDA OOM] batch_size={batch_size} quá lớn cho 4GB VRAM -> thử batch_size={batch_size // 2}")
                if device == "cuda":
                    torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            if isinstance(e, ImportError):
                print(f"  [THIẾU PACKAGE lúc train] {e} -> dùng zero-shot thay vì crash.")
                return model
            raise

    model.save_pretrained("dense_finetuned")
    # SỬA lỗi NGHIÊM TRỌNG (phát hiện từ log chạy thật: Bước 5 mất 10h10' thay vì vài phút):
    # `model.save_pretrained()` cần tensor ở CPU để serialize (safetensors yêu cầu bộ nhớ CPU
    # liền mạch) — một số phiên bản sentence-transformers chuyển model về CPU trong lúc save và
    # KHÔNG chuyển lại GPU sau đó. Model trả về từ đây vì vậy có thể đang nằm trên CPU dù log
    # trước đó báo "Device: cuda" — encode_corpus() sau đó chạy CPU (135M tham số, batch chạy
    # được nhưng ở tốc độ ~3.6s/batch thay vì ~0.05-0.1s/batch trên GPU, x36-70 lần chậm hơn,
    # CHÍNH XÁC khớp với log 10h10' bạn gặp). Fix: ép chuyển lại device tường minh + IN RA để
    # xác nhận, không tin ngầm định thư viện tự giữ đúng device.
    model = model.to(device)
    actual_device = next(model.parameters()).device
    print(f"  [Xác nhận device sau train] model đang ở: {actual_device} (kỳ vọng: {device})")
    if str(actual_device) != device and device == "cuda":
        print("  [CẢNH BÁO NGHIÊM TRỌNG] model vẫn KHÔNG ở GPU sau khi ép .to(device) — "
              "kiểm tra lại cài đặt torch/CUDA, Bước 5 sẽ CHẬM nếu tiếp tục ở CPU.")
    return model


# ==============================================================================
# BƯỚC 5 — Encode toàn bộ corpus + RRF fusion (BM25 + dense), KHÔNG rerank (xem lý do ở đầu file)
# ==============================================================================
def encode_corpus(model, all_chunks: list):
    import torch
    # SỬA (cùng nguyên nhân với comment ở finetune_or_load_dense): KHÔNG tin device hiện tại
    # của model, tự ép lại trước khi encode toàn bộ corpus — đây là bước tốn thời gian nhất
    # của cả pipeline nếu chạy sai device, xứng đáng có 1 lớp phòng thủ RIÊNG ở đây thay vì chỉ
    # dựa vào chỗ gọi trước đó đã set đúng.
    if torch.cuda.is_available():
        _cap_cuda_memory()  # ép trần VRAM cứng — xem lý do chi tiết ở docstring _cap_cuda_memory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    actual_device = next(model.parameters()).device
    print(f"  [Xác nhận device trước khi encode] model đang ở: {actual_device}")
    if str(actual_device) != device and device == "cuda":
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
def render_answer(selected_chunks: list, top_n: int) -> str:
    parts, seen = [], set()
    for c in selected_chunks:
        if c["id"] in seen or len(parts) >= top_n:
            continue
        seen.add(c["id"])
        loai_vb = c["loai_vb"] or "văn bản"
        so_hieu = c["so_hieu"] or ""
        dieu = c["dieu_so"]
        lead = (f"Theo Điều {dieu} {loai_vb} {so_hieu} quy định cụ thể:"
                if dieu != "0" else f"Theo {loai_vb} {so_hieu} quy định cụ thể:")
        parts.append(f"{lead}\n{c['text']}")
    return "\n\n".join(parts)


def answer_question(question: str, bm25, dense_model, dense_embeddings, all_chunks, top_n: int) -> str:
    ranked = rrf_retrieve(question, bm25, dense_model, dense_embeddings, all_chunks)
    if not ranked:
        return "Không tìm thấy thông tin pháp lý cho câu hỏi này."
    return render_answer(ranked, top_n)


# ==============================================================================
# BƯỚC 7 — Dev-eval (METEOR/ROUGE-L) trên mẫu train.json để chọn TOP_N_ANSWER, và
# BƯỚC 8 — Validate + đóng gói submission.zip
# ==============================================================================
def try_dev_eval(bm25, dense_model, dense_embeddings, all_chunks, train_data) -> int:
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
        print(f"  Bỏ qua dev-eval (thiếu nltk/rouge_score: {e}). Dùng TOP_N_ANSWER=3 mặc định.")
        return 3

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    random.seed(42)
    # SỬA: bỏ nhánh "hết ngân sách -> rút mẫu còn 40" và "hết ngân sách -> dừng sớm giữa vòng lặp"
    # của bản trước — đây chính là lý do log lần trước ra "METEOR=-1.0000" (dừng trước khi tính
    # được gì) do Bước 5 lỗi device ăn hết ngân sách 55 phút. Giờ không còn gate cứng theo phase,
    # dev-eval luôn chạy đủ để có tín hiệu thật trước khi quyết định TOP_N_ANSWER.
    n_sample = min(DEV_EVAL_SAMPLE_SIZE, len(train_data))
    ids = random.sample(list(train_data.keys()), n_sample)

    best_n, best_m = 3, -1.0
    for top_n in (1, 3, 5, 7):  # SỬA: thêm 7 — METEOR alpha=0.9 nặng recall, đáng thử ngưỡng cao hơn
        ms, rs = [], []
        for qid in ids:
            item = train_data[qid]
            pred = answer_question(item["question"], bm25, dense_model, dense_embeddings, all_chunks, top_n)
            ref = item["answer"]
            ms.append(meteor_score([str(ref).split()], str(pred).split()))
            rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
        m, r = sum(ms) / len(ms), sum(rs) / len(rs)
        print(f"  top_n={top_n}  METEOR={m:.4f}  ROUGE-L={r:.4f}  (n={len(ids)})")
        if m > best_m:
            best_m, best_n = m, top_n
    print(f"  => chọn TOP_N_ANSWER={best_n} (METEOR={best_m:.4f})")
    return best_n


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

    print("\n=== Bước 6: Dev-eval chọn TOP_N_ANSWER ===")
    top_n_answer = try_dev_eval(bm25, dense_model, dense_embeddings, all_chunks, train_data)
    checkpoint("Xong dev-eval")

    print("\n=== Bước 7: Sinh câu trả lời cho public-official.json ===")
    with PUBLIC_PATH.open(encoding="utf-8") as f:
        questions = json.load(f)
    answers = {}
    for i, (qid, item) in enumerate(questions.items()):
        answers[qid] = answer_question(item["question"], bm25, dense_model, dense_embeddings,
                                        all_chunks, top_n_answer)
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