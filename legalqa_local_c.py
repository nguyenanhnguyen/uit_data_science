#!/usr/bin/env python
"""
legalqa_dual_encoder.py — LegalQA (UIT DSC2026 Task 2), kiến trúc v7: xếp hạng + hiệu năng.

Phát hiện định hướng nghiên cứu: Recall@100 = 100% ở MỌI cấu hình đo được trên log thật —
retrieval đã bão hoà hoàn toàn, nút thắt còn lại nằm ở XẾP HẠNG + QUYẾT ĐỊNH SỐ LƯỢNG,
không phải "tìm đâu thấy". Bản này KHÔNG thêm retriever mới, cải thiện đúng chỗ đó +
tăng tốc lặp thử nghiệm:
  1. BM25 dùng tách từ tiếng Việt thật (underthesea, tự đo tốc độ + lùi về regex nếu quá
     chậm) — sửa bẫy từ ghép "hợp_đồng" (corpus) vs "hợp đồng" (câu hỏi) không khớp token.
  2. Hard-negative CÙNG VĂN BẢN cho reranker (thay BM25 top-60 chung chung) — nhắm đúng vào
     lỗi đã đo: reranker cũ học trên negative quá dễ, không phân biệt được các Điều cùng
     văn bản.
  3. Tái dùng checkpoint (2 dense encoder + reranker) nếu fingerprint khớp cấu hình hiện
     tại — train chiếm ~32% thời gian 1 lần chạy đầy đủ, lãng phí nếu chỉ thử lại Bước 6/7.
  4. Nâng ENCODE_BATCH_SIZE — encode corpus chiếm ~32% thời gian, 16GB/thẻ Kaggle còn dư.

Giữ nguyên mọi kỹ thuật đã xác nhận có ích: 2 dense encoder (BAAI/bge-m3 + intfloat/
multilingual-e5-large) fine-tune song song 2 GPU (subprocess), fusion RRF 3 kênh, fine-tune
reranker (loss logistic + dừng sớm), lưới an toàn so sánh không-rerank/zero-shot/fine-tuned/
RRF-gaps ở Bước 6, nhãn Task 1 (LegalIR), LoRA + optimizer 8-bit cho VRAM thấp, tự nhận diện
môi trường Kaggle/máy cá nhân, seed toàn cục, sổ thí nghiệm.

Bản .py TRÍCH XUẤT Y HỆT logic của legalqa_kaggle_t4x2.ipynb (không viết lại tay, tránh
lệch giữa 2 bản) — chạy Ở CẤP MODULE, không gói trong def main()/không thụt lề gì (tránh
lỗi IndentationError đã gặp thật khi thụt lề làm hỏng chuỗi worker_code nhiều dòng).

CÁCH DÙNG (đặt cạnh train.json, public-official.json, selected-contexts/, và
legalir_train.json của Task 1 nếu có — không có thì tự lùi về chỉ nhãn citation):
    pip install -q -U sentence-transformers datasets "accelerate>=1.1.0" nltk rouge_score sentencepiece peft underthesea
    pip install -q -U bitsandbytes   # tuỳ chọn — không cài được cũng không sao
    python legalqa_dual_encoder.py

KHÔNG dùng ký hiệu `!pip install` (cú pháp riêng của Jupyter) — cài thư viện bằng lệnh
pip ở trên TRƯỚC khi chạy file này.
"""

# Cell 2: Đường dẫn và tham số
import os

# BẢN SỬA (log lỗi thật: chạy trên máy cá nhân RTX 2050 4GB nhưng CONTEXT_DIR trước đây cứng
# đường dẫn Kaggle /kaggle/input/... -> FileNotFoundError): tự nhận diện MÔI TRƯỜNG thay vì
# cứng 1 kiểu đường dẫn — /kaggle/input CHỈ tồn tại thật trên Kaggle, nên dùng chính nó làm
# điều kiện phát hiện. Nhờ vậy CÙNG MỘT nguồn code chạy đúng trên cả 2 nơi (notebook Kaggle
# tự lấy đường dẫn Kaggle, .py trên máy cá nhân tự lấy đường dẫn CẠNH SCRIPT — đúng quy ước
# HERE-relative của legalqa_local.py, dữ liệu đặt cùng thư mục file .py).
IS_KAGGLE = os.path.isdir("/kaggle/input")

if IS_KAGGLE:
    DATA_DIR = "/kaggle/input/datasets/anhnguyen7508/uit-data-science-dataset/"
    CONTEXT_DIR = os.path.join(DATA_DIR, "selected-contexts/selected-contexts/")
    TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
    WARMUP_PATH = os.path.join(DATA_DIR, "warmup.json")
    PUBLIC_PATH = os.path.join(DATA_DIR, "public-official.json")
    TASK1_TRAIN_PATH = os.path.join(DATA_DIR, "legalir_train.json")  # train.json của Task 1
    # (LegalIR) — nhãn document-level SẠCH, phủ 100% câu hỏi (xem build_task1_pairs ở Cell 7).
    # Cần tự thêm ~1MB file này vào Kaggle Dataset nếu chưa có.
    # /kaggle/input CHỈ ĐỌC. Mọi thứ ghi ra phải nằm ở /kaggle/working (được lưu khi
    # "Save Version" — đây là nơi submission.zip phải nằm) hoặc /kaggle/temp (KHÔNG được
    # lưu, mất khi session kết thúc — dùng cho cache/model tải về, đỡ tốn quota output).
    OUT_DIR = "/kaggle/working"
    CACHE_DIR = "/kaggle/temp/legalqa_cache"
else:
    # Máy cá nhân (hoặc bất kỳ máy nào không phải Kaggle): đặt file .py CẠNH train.json,
    # public-official.json, selected-contexts/ — đúng quy ước của legalqa_local.py, không có
    # trần thời gian phiên nào để lo (khác Kaggle) nên OUT_DIR/CACHE_DIR cũng nằm ngay cạnh
    # script, dễ tìm dễ dọn.
    HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    DATA_DIR = HERE
    CONTEXT_DIR = os.path.join(HERE, "selected-contexts")
    TRAIN_PATH = os.path.join(HERE, "train.json")
    WARMUP_PATH = os.path.join(HERE, "warmup.json")
    PUBLIC_PATH = os.path.join(HERE, "public-official.json")
    TASK1_TRAIN_PATH = os.path.join(HERE, "legalir_train.json")  # train.json của Task 1,
    # đặt cạnh script nếu có (xem build_task1_pairs ở Cell 7) — không có thì tự bỏ qua êm.
    OUT_DIR = HERE
    CACHE_DIR = os.path.join(HERE, "cache")

HF_CACHE_DIR = os.path.join(CACHE_DIR, "hf")
NLTK_CACHE_DIR = os.path.join(CACHE_DIR, "nltk_data")
TRAINER_TMP_DIR = os.path.join(CACHE_DIR, "trainer_tmp")
for _d in (OUT_DIR, HF_CACHE_DIR, NLTK_CACHE_DIR, TRAINER_TMP_DIR):
    os.makedirs(_d, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_CACHE_DIR, "hub"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Tham số
CHUNK_SIZE = 512      # không sử dụng — chunk theo Điều (xem Bước 1), giữ lại đúng như đề bài
TOP_K_RETRIEVE = 100  # số ứng viên lấy ra sau RRF fusion (BM25 + dense)
TOP_K_RERANK = 5      # trần trên cho số Điều đưa vào 1 câu trả lời (top_n tĩnh VÀ trần adaptive-k)
USE_FINETUNE = True   # bật mặc định — có đủ VRAM (16GB/thẻ Kaggle, hoặc OOM-backoff tự lùi
                       # trên máy yếu hơn) thì fine-tune, không cần đắn đo trước. Đặt False nếu
                       # muốn chạy thử nhanh hoặc đang tiết kiệm quota GPU trên Kaggle.

# Kiến trúc "mạnh nhất" — 2 dense encoder khác họ, fine-tune SONG SONG trên 2 GPU riêng (nếu
# có; tự lùi về tuần tự nếu chỉ 1 GPU — xem Bước 4), fusion RRF 3 kênh với BM25, thay cho 1
# bi-encoder 135M tự train trước đây. Không có nhãn document-level của Task 1 nên train HOÀN
# TOÀN từ nhãn citation của Task 2 — chỉ đổi SỐ LƯỢNG và ĐỘ MẠNH encoder, không cần dữ liệu ngoài.
BASE_DENSE_MODEL_A = "BAAI/bge-m3"                        # ~568M, đa ngôn ngữ, không cần tiền tố
BASE_DENSE_MODEL_B = "intfloat/multilingual-e5-large"     # ~560M, CẦN tiền tố "query: "/"passage: "
                                                            # — bẫy kinh điển: quên tiền tố thì
                                                            # recall tụt mà KHÔNG có lỗi nào bắn ra
                                                            # (embedding vẫn ra số, chỉ lệch hệ toạ độ).
DENSE_MAX_SEQ_LEN = 256
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")   # 1 thư mục duy nhất — Kaggle: giữ lại
                                                          # khi Save Version, tự tải về; máy cá
                                                          # nhân: nằm cạnh script như mọi output khác.
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
MIN_TRAIN_PAIRS = 50
MAX_TRAIN_EXAMPLES = 9000   # BẢN SỬA: nâng 3000 -> 9000. Với USE_TASK1_LABELS=True số
                             # positive pair tăng từ ~3.300 (citation) lên ~10.000, để trần
                             # 3000 thì phần nhãn Task 1 vừa thêm gần như bị vứt đi. Không
                             # sợ vượt giờ: Bước 4 vẫn time-box theo FINETUNE_TIME_BUDGET_SEC
                             # (đo tốc độ vài step đầu rồi tự tính max_steps).
N_NEG_PER_ROW = 2

# BẢN SỬA (theo yêu cầu — chạy trên máy cá nhân, không cần đắn đo ngân sách còn lại): batch
# khởi điểm GIỮ NGUYÊN mức đã tối ưu cho GPU nhiều VRAM (Kaggle T4 16GB); trên máy VRAM nhỏ
# hơn (vd RTX 2050 4GB), OOM-backoff đã có sẵn ở mọi bước (Bước 4/5/6/7/5b) tự lùi batch khi
# OOM thật xảy ra — KHÔNG cần hạ tay trước, đúng triết lý "dùng tối đa tài nguyên, chỉ lùi khi
# thật sự hết" đã áp dụng xuyên suốt từ legalqa_local.py.
TRAIN_BATCH_SIZE = 64        # batch HIỆU DỤNG (số in-batch negative)
TRAIN_MINI_BATCH_SIZE = 32   # batch THẬT mỗi forward — OOM-backoff tự giảm nếu máy yếu hơn Kaggle.
ENCODE_BATCH_SIZE = 384     # BẢN SỬA: nâng từ 256 -- log thật: encode corpus chiếm
                              # ~121/380 phút (32%) của 1 lần chạy đầy đủ, 16GB/thẻ Kaggle
                              # còn dư -- batch lớn hơn tận dụng tốt hơn (OOM-backoff tự lùi
                              # nếu máy yếu hơn dự kiến).
RERANK_SUBBATCH = 64         # batch xử lý mỗi lần forward reranker (568M) — xử lý theo lô thay
                              # vì lô vừa phải luôn nhanh hơn 1 lô khổng lồ (padding ít hơn,
                              # không nghẽn băng thông bộ nhớ) — xem Cell 11.

# BẢN SỬA: TIME_BUDGET chỉ thật sự cần trên Kaggle (trần phiên GPU ~9-12h). Máy cá nhân KHÔNG
# có giới hạn phiên nào — đặt trần RẤT RỘNG (không phải vô hạn, để tránh treo vĩnh viễn nếu có
# bug logic nào đó) thay vì ép chạy nhanh/cắt ngắn không cần thiết.
TIME_BUDGET_SEC = (8 * 3600) if IS_KAGGLE else (48 * 3600)
FINETUNE_TIME_BUDGET_SEC = (3 * 3600) if IS_KAGGLE else (16 * 3600)
DEV_EVAL_SAMPLE_SIZE = 300

# BẢN SỬA (tái lập được kết quả + ablation warmup + sổ thí nghiệm — theo phân tích
# chênh lệch điểm .py-vs-Kaggle, xem PHAN_TICH_KY_THUAT.md): trước đây random.seed(42) chỉ
# đặt ngay trước lúc lấy mẫu dev-eval (Bước 6) — random.sample/random.choice ở Bước 4 (chọn
# tập con để fine-tune, chọn hard-negative dự phòng) chạy TRƯỚC đó với random state KHÔNG
# seed, nên 2 lần chạy cùng code vẫn fine-tune trên 2 tập con khác nhau. SEED được set_all_seeds()
# NGAY SAU khi import xong (Cell 4) — trước Bước 3/4 — để tái lập được. USE_WARMUP cho phép
# ablation có/không warmup.json ở CÙNG seed. EXPERIMENT_LOG_PATH nằm trong OUT_DIR — trên
# Kaggle được giữ lại khi "Save Version" (khác cache/ ở /kaggle/temp, mất khi session kết
# thúc); trên máy cá nhân nằm cạnh script như submission.zip.
# CÂU KẾT — xem docstring render_answer() ở Cell 11 để biết số đo đầy đủ (+4,8 điểm METEOR
# so với không có câu kết, đo trên 501 câu dev, xác nhận bằng split-half).
CONCL = "echo2"          # none | echo | echo2

# NHÃN HUẤN LUYỆN — xem docstring build_task1_pairs() ở Cell 7.
# Task 1 (LegalIR) dùng CÙNG corpus 8.532 văn bản (đã đối chiếu md5) và có 7.000 nhãn
# question -> document_id SẠCH, phủ 100%. Nhãn citation của Task 2 chỉ phân giải được
# 47,8% câu. Hai tập câu hỏi gần như rời nhau (21/7000 trùng) nên KHÔNG rò rỉ.
# Đây vẫn là dữ liệu BTC, không phải dữ liệu ngoài.
USE_TASK1_LABELS = True

SEED = 42
USE_WARMUP = True   # đặt False để ablation: chỉ dùng train.json, không gộp warmup.json
EXPERIMENT_LOG_PATH = os.path.join(OUT_DIR, "experiment_log.jsonl")

# BẢN SỬA (kết quả thật: dual-encoder 0.5215/0.4829 chỉ nhích rất ít so với single-encoder
# 0.5199/0.4806 dù retrieval mạnh hơn nhiều -> retrieval không còn là nút thắt chính, reranker
# ZERO-SHOT giờ nhiều khả năng là trần chặn điểm. Fine-tune reranker trên chính nhãn citation
# Task 2 -- tái dùng `rows` đã build cho Bước 4, KHÔNG cần dữ liệu thêm.
USE_RERANKER_FINETUNE = True
RERANKER_BASE = "AITeamVN/Vietnamese_Reranker"
RERANKER_FT_TIME_BUDGET_SEC = (60 * 60) if IS_KAGGLE else (6 * 3600)
RERANKER_FT_BATCH_SIZE = 8              # số CÂU HỎI/batch (mỗi câu có 1 positive + N_NEG_PER_ROW
                                         # negative -> batch thật cho reranker lớn hơn số này)
RERANKER_FT_LR = 1e-5
RERANKER_FT_MARGIN = 1.0                # margin ranking loss: điểm(positive) phải > điểm(negative)
                                         # + margin -- không cần thang điểm chuẩn hoá, chỉ cần đúng
                                         # THỨ TỰ, ổn định hơn BCE/MSE cho reranker logit thô.

# BẢN SỬA (log lỗi thật: torch.AcceleratorError OOM ngay ở lần optimizer.step() ĐẦU TIÊN,
# trước khi kịp chạy batch nào — full fine-tune AdamW cho model ~568M cần khoảng 6-7GB CHỈ
# RIÊNG optimizer state (2 buffer fp32/tham số), không phụ thuộc batch size. Trên GPU 4GB,
# KHÔNG batch nào nhỏ tới đâu cũng không đủ — đây là giới hạn vật lý, không phải cấu hình sai.
# LoRA (chỉ train 1 phần rất nhỏ tham số, đóng băng phần còn lại) không phải "cho nhanh hơn"
# mà là ĐIỀU KIỆN BẮT BUỘC để fine-tune được model cỡ này trên 4GB — đồng thời PHÙ HỢP HƠN
# về phương pháp với lượng dữ liệu nhỏ hiện có (3.500-9.000 câu là rất ít so với 568M tham
# số, full fine-tune có rủi ro overfit/quên kiến thức gốc thật sự). Trên Kaggle (nhiều VRAM),
# GIỮ NGUYÊN full fine-tune như cũ — không đổi kết quả 0.5526/0.4817 đã có kiểm chứng.
_total_vram_gb = 0.0
try:
    import torch as _torch_probe
    if _torch_probe.cuda.is_available():
        _total_vram_gb = _torch_probe.cuda.get_device_properties(0).total_memory / (1024 ** 3)
except Exception:
    pass
LOW_VRAM_MODE = (not IS_KAGGLE) and (_total_vram_gb > 0) and (_total_vram_gb < 10)
USE_LORA = LOW_VRAM_MODE        # LoRA cho CẢ dense encoder LẪN reranker khi VRAM thấp
USE_8BIT_OPTIM = LOW_VRAM_MODE  # optimizer AdamW 8-bit (bitsandbytes) khi VRAM thấp — cộng
                                 # dồn với LoRA, không thay thế; tự lùi về AdamW thường êm ái
                                 # nếu bitsandbytes không cài được (xem Cell 1).
LORA_R = 16          # rank — 16 là điểm cân bằng phổ biến, đủ biểu đạt cho fine-tune domain
LORA_ALPHA = 32      # thường đặt = 2 * LORA_R
LORA_DROPOUT = 0.05

print(f"Môi trường: {'Kaggle' if IS_KAGGLE else 'máy cá nhân (không phải Kaggle)'}")
print(f"DATA_DIR  = {DATA_DIR}")
print(f"OUT_DIR   = {OUT_DIR}")
print(f"CACHE_DIR = {CACHE_DIR}" + ("  (tạm, mất khi session kết thúc)" if IS_KAGGLE else ""))
print(f"LOW_VRAM_MODE={LOW_VRAM_MODE} (VRAM={_total_vram_gb:.1f}GB) -> USE_LORA={USE_LORA}, USE_8BIT_OPTIM={USE_8BIT_OPTIM}")

# BẢN SỬA (research mới — retrieval đã bão hoà: Recall@100=100% ở MỌI cấu hình đo được, xem
# PHAN_TICH_KY_THUAT.md — nút thắt còn lại nằm ở XẾP HẠNG/CHỌN SỐ LƯỢNG, không phải "tìm
# đâu thấy" nữa). Bốn thay đổi, đều nhắm đúng vào đó thay vì thêm retriever:
USE_VI_TOKENIZER = True
# BM25 hiện tách từ bằng regex thô (mỗi âm tiết 1 token) — corpus tiếng Việt có TỪ GHÉP
# ("hợp_đồng", "quyết_định") mà câu hỏi gõ tay thường KHÔNG nối gạch dưới ("hợp đồng") ->
# 2 token khác nhau, BM25 mất tín hiệu dù cùng nghĩa. Bật tách từ thật (underthesea) cho cả
# corpus lẫn câu hỏi. CÓ TỰ KIỂM TỐC ĐỘ (đo 200 chunk đầu, ước lượng tổng thời gian) — nếu
# chiếu ra quá 40 phút cho toàn corpus thì TỰ LÙI về regex thô, không ép chạy chậm vô hạn.

REUSE_CHECKPOINT_IF_EXISTS = True
# Log thật: train 2 dense encoder (~90 phút) + reranker (~29 phút) chiếm ~120/380 phút (32%)
# một lần chạy đầy đủ — lãng phí nếu chỉ muốn thử lại Bước 6/7 (đổi adaptive-k, câu kết...)
# mà không đổi gì ở dữ liệu train. Nếu checkpoint trong CHECKPOINT_DIR khớp ĐÚNG fingerprint
# (số cặp train, SEED, model gốc, USE_LORA...) với cấu hình hiện tại, TẢI LẠI thay vì train
# lại từ đầu. Fingerprint LỆCH dù chỉ 1 chi tiết -> train lại bình thường, không bao giờ dùng
# nhầm checkpoint cũ không khớp.

print(f"USE_VI_TOKENIZER={USE_VI_TOKENIZER} · REUSE_CHECKPOINT_IF_EXISTS={REUSE_CHECKPOINT_IF_EXISTS}")


# Cell 3: Kiểm tra GPU (mong đợi 2x Tesla T4) + tiện ích thời gian
import time
import torch

N_GPU = torch.cuda.device_count()
print(f"Số GPU thấy được: {N_GPU}")
for i in range(N_GPU):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} — {p.name}, {p.total_memory/1024**3:.1f} GB")

if N_GPU == 0:
    DEVICES = ["cpu"]
    print("[CẢNH BÁO] Không thấy GPU nào — kiểm tra Settings > Accelerator = GPU T4 x2. "
          "Sẽ chạy CPU, RẤT chậm cho Bước 4/5/6/7.")
elif N_GPU == 1:
    DEVICES = ["cuda:0"]
    print("[CẢNH BÁO] Chỉ thấy 1 GPU — vẫn chạy được nhưng KHÔNG tận dụng song song 2 thẻ "
          "ở Bước 5/6/7. Kiểm tra Settings > Accelerator = GPU T4 x2 nếu muốn đủ 2 thẻ.")
else:
    DEVICES = [f"cuda:{i}" for i in range(N_GPU)]
    print(f"OK — sẽ dùng song song {DEVICES} ở Bước 5 (encode corpus) và Bước 6/7 (rerank).")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

_START_TIME = time.time()

def elapsed() -> float:
    return time.time() - _START_TIME

def remaining() -> float:
    return TIME_BUDGET_SEC - elapsed()

def checkpoint(label: str) -> None:
    print(f"[{elapsed()/60:6.1f} phút] {label}  (còn lại ~{remaining()/60:.1f} phút trong ngân sách)")

checkpoint("Bắt đầu")

# Cell 4: Import chung + hằng số regex cho chunk theo Điều
import re
import json
import math
import random
import zipfile
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

DIEU_RE = re.compile(r"^[ \t]*Điều\s+(\d+)[a-zđA-ZĐ]?[\.\s]", re.MULTILINE)
SO_HEADER_RE = re.compile(r"Số\s*[:：]\s*([0-9A-Za-zĐđ/\-]+)")
SO_HIEU_RE = re.compile(r"\d{1,6}[A-Za-z]{0,3}/(?:\d{4}/)?[A-Za-zĐđ]{2,10}(?:-[A-Za-zĐđ]{2,10})?")
LOAI_VB_CANON = ["Thông tư liên tịch", "Nghị định", "Luật", "Thông tư", "Quyết định",
                 "Pháp lệnh", "Nghị quyết", "Bộ luật", "Chỉ thị"]
LOAI_PATTERN = re.compile("(" + "|".join(re.escape(x) for x in LOAI_VB_CANON) + ")", re.IGNORECASE)
DIEU_CITATION_RE = re.compile(r"Điều\s+(\d+)\s*[a-zđA-ZĐ]?\b")
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
_DIEU_PREFIX_STRIP_RE = re.compile(r"^\s*Điều\s+\d+[a-zđA-ZĐ]?\.?\s*", re.IGNORECASE)


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


# BẢN SỬA (research mới — BM25 đang mất tín hiệu vì từ ghép trong corpus có gạch dưới
# "hợp_đồng" trong khi câu hỏi gõ tay "hợp đồng" -- 2 token khác nhau dù cùng nghĩa): thử
# tách từ tiếng Việt thật (underthesea) thay cho regex mỗi-âm-tiết-1-token. `_VI_SEGMENT_OK`
# quyết định 1 LẦN ở Bước 2 (sau speed-check trên mẫu) rồi dùng NHẤT QUÁN cho cả corpus lẫn
# mọi câu hỏi về sau — BẮT BUỘC nhất quán, dùng 2 kiểu tách từ khác nhau cho corpus và câu
# hỏi sẽ làm BM25 hỏng hoàn toàn (token không bao giờ khớp).
try:
    from underthesea import word_tokenize as _vi_word_tokenize
except ImportError:
    _vi_word_tokenize = None
_VI_SEGMENT_OK = False


def _tokenize_regex(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


def tokenize_simple(text: str) -> list:
    if _VI_SEGMENT_OK and _vi_word_tokenize is not None:
        try:
            return [t.lower() for t in _vi_word_tokenize(text)]
        except Exception:
            return _tokenize_regex(text)
    return _tokenize_regex(text)


def norm_so_hieu(s: str) -> str:
    return s.strip().upper()

# BẢN SỬA: seed toàn cục NGAY SAU khi import xong (trước Bước 3 ở Cell 7, trước Bước 4 ở
# Cell 8 — hai nơi duy nhất gọi random.sample/random.choice) — xem giải thích đầy đủ ở Cell 2.
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_all_seeds(SEED)
print(f"  Đã seed toàn cục với SEED={SEED} (random/numpy/torch) — trước mọi lời gọi random "
      f"ở Bước 3/4, để nhiều lần chạy cùng code fine-tune trên cùng 1 tập con, tái lập được.")

# Cell 5: Bước 1 — Chunk corpus theo Điều (neo đầu dòng — tránh rách nội dung khi 1 Điều
# trích dẫn Điều khác trong thân bài) + trích so_hieu/loai_vb từ NỘI DUNG (không phải tên file)
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


def load_corpus(contexts_dir) -> list:
    contexts_dir = Path(contexts_dir)
    if not contexts_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy {contexts_dir} — kiểm tra lại CONTEXT_DIR ở Cell 2.")
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


print("=== Bước 1: Chunk corpus ===")
all_chunks = load_corpus(CONTEXT_DIR)
checkpoint("Xong chunking")

# Cell 6: Bước 2 — BM25 tự viết bằng numpy (inverted index vector hoá — nhanh trên corpus lớn)
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

        self.inverted: dict = {}
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


print("=== Bước 2: BM25 index ===")
if USE_VI_TOKENIZER and _vi_word_tokenize is not None:
    print("  Đo tốc độ tách từ tiếng Việt (underthesea) trên 200 chunk mẫu...")
    _sample = all_chunks[:200] if len(all_chunks) >= 200 else all_chunks
    _t0 = time.time()
    _ok = True
    for _c in _sample:
        try:
            _vi_word_tokenize(_c["text"][:2000])  # cắt bớt chunk siêu dài để đo ổn định
        except Exception:
            _ok = False
            break
    _elapsed_sample = time.time() - _t0
    _projected_min = (_elapsed_sample / max(len(_sample), 1)) * len(all_chunks) / 60
    print(f"  {_elapsed_sample:.1f}s cho {len(_sample)} chunk -> ước lượng {_projected_min:.1f} phút "
          f"cho toàn bộ {len(all_chunks)} chunk.")
    if _ok and _projected_min <= 40:
        _VI_SEGMENT_OK = True
        print("  -> BẬT tách từ tiếng Việt cho BM25 (trong ngân sách 40 phút).")
    else:
        print(f"  -> {'lỗi lúc tách từ' if not _ok else f'quá chậm (ước lượng {_projected_min:.1f} phút > 40 phút)'} "
              f"-> LÙI về tách từ regex thô.")
elif USE_VI_TOKENIZER:
    print("  USE_VI_TOKENIZER=True nhưng underthesea chưa cài được -> dùng tách từ regex thô.")

_t0 = time.time()
tokenized = [tokenize_simple(f"{c.get('loai_vb','')} {c['text']}") for c in all_chunks]
print(f"  Tách từ xong corpus: {time.time()-_t0:.0f}s "
      f"({'underthesea' if _VI_SEGMENT_OK else 'regex'})")
bm25 = BM25(tokenized)
checkpoint("Xong BM25 index")

# Cell 7: Bước 3 — Sinh nhãn (question -> chunk) từ citation trong train.json (+ warmup.json
# nếu USE_WARMUP=True, cùng schema — gộp thêm dữ liệu train, KHÔNG gộp vào mẫu dev-eval để
# tránh lẫn chất lượng nhãn chưa kiểm chứng vào lúc CHỌN cấu hình cuối cùng)
def extract_citations(answer) -> list:
    # answer có thể KHÔNG phải string (đã gặp thật: warmup.json có answer kiểu list ở một số
    # câu, khác train.json toàn string) — bỏ qua câu đó thay vì crash cả pipeline.
    if not isinstance(answer, str):
        return []
    out = []
    for m in DIEU_CITATION_RE.finditer(answer):
        window = answer[m.end(): m.end() + 60]
        so_m = SO_HIEU_RE.search(window)
        if so_m and so_m.start() <= 40:
            out.append((m.group(1), so_m.group(0)))
    return out


def build_train_pairs(train_data: dict, all_chunks: list):
    so_hieu_index = {}
    for c in all_chunks:
        if c["so_hieu"] and c["dieu_so"] != "0":
            so_hieu_index.setdefault((c["dieu_so"], norm_so_hieu(c["so_hieu"])), c["id"])

    positive = {}
    n_skipped_type = 0
    for qid, item in train_data.items():
        if not isinstance(item.get("answer"), str):
            n_skipped_type += 1
            continue
        for dieu, so_hieu in extract_citations(item["answer"]):
            key = (dieu, norm_so_hieu(so_hieu))
            if key in so_hieu_index:
                positive[qid] = so_hieu_index[key]
                break
    if n_skipped_type:
        print(f"  [CẢNH BÁO] {n_skipped_type} câu có answer KHÔNG phải string -> bỏ qua khi "
              f"sinh nhãn, không tính vào positive pairs.")
    chunk_by_id = {c["id"]: c for c in all_chunks}
    return positive, chunk_by_id


print("=== Bước 3: Sinh nhãn từ train.json" + (" + warmup.json" if USE_WARMUP else "") + " ===")
with open(TRAIN_PATH, encoding="utf-8") as f:
    train_data = json.load(f)
print(f"  train.json: {len(train_data)} câu")

# warmup.json: gộp thêm CHỈ KHI USE_WARMUP=True VÀ file tồn tại VÀ đúng schema
# {qid: {"question": str, "answer": str}} — lọc chặt cả kiểu dữ liệu. KHÔNG gộp vào mẫu
# dev-eval ở Bước 6 — dev-eval chỉ dùng train.json gốc để giữ tín hiệu chọn cấu hình đáng tin
# (xem phần 2.2 trong PHAN_TICH_KY_THUAT.md). USE_WARMUP=False cho phép ablation có kiểm soát.
train_data_for_pairs = dict(train_data)
n_warmup_used = 0
if USE_WARMUP and os.path.exists(WARMUP_PATH):
    try:
        with open(WARMUP_PATH, encoding="utf-8") as f:
            warmup_data = json.load(f)
        n_bad_type = 0
        for qid, item in warmup_data.items():
            if not isinstance(item, dict):
                n_bad_type += 1
                continue
            q, a = item.get("question"), item.get("answer")
            if isinstance(q, str) and isinstance(a, str):
                train_data_for_pairs[f"warmup_{qid}"] = {"question": q, "answer": a}
                n_warmup_used += 1
            else:
                n_bad_type += 1
        print(f"  warmup.json: {len(warmup_data)} câu, {n_warmup_used} câu đúng schema -> gộp thêm"
              + (f", {n_bad_type} câu sai kiểu -> bỏ qua." if n_bad_type else ".")
              + " KHÔNG gộp vào mẫu dev-eval/Recall@k ở Bước 6.")
    except Exception as e:
        print(f"  [CẢNH BÁO] Có WARMUP_PATH nhưng đọc lỗi ({e}) -> bỏ qua, chỉ dùng train.json.")
elif USE_WARMUP:
    print(f"  USE_WARMUP=True nhưng không thấy {WARMUP_PATH} -> chỉ dùng train.json.")
else:
    print(f"  USE_WARMUP=False -> chỉ dùng train.json (bỏ qua warmup.json dù có tồn tại).")

train_positive, chunk_by_id = build_train_pairs(train_data_for_pairs, all_chunks)
print(f"  Positive pairs: {len(train_positive)}/{len(train_data_for_pairs)}")
# ---------------------------------------------------------------------------
# BẢN SỬA: thêm nhãn của TASK 1 (LegalIR) — nguồn giám sát mạnh hơn hẳn citation
# ---------------------------------------------------------------------------
def build_task1_pairs(task1_path, all_chunks, bm25, seen_questions: set):
    """{qid: chunk_id} suy từ nhãn document_id của Task 1.

    Vì sao đáng đổi: nhãn citation của Task 2 chỉ phân giải được **47,8%** câu (3.349/7.000)
    và chỉ khi đáp án có số hiệu văn bản viết rõ. Task 1 cho **7.000 câu, phủ 100%**, nhãn
    do BTC gán ở mức document — sạch hơn nhiều. Corpus của hai task TRÙNG KHÍT (8.532 file,
    đã đối chiếu md5), còn câu hỏi thì gần như rời nhau (chỉ 21/7.000 trùng) nên dùng nhãn
    Task 1 để train KHÔNG rò rỉ vào dev-eval của Task 2. Vẫn là dữ liệu BTC.

    Nhãn Task 1 ở mức DOCUMENT còn ta cần mức CHUNK (Điều). Giám sát yếu: trong đúng văn
    bản gold, lấy Điều mà BM25 chấm cao nhất cho câu hỏi đó. Không hoàn hảo, nhưng sai ở
    mức "đúng văn bản, lệch Điều" — nhẹ hơn nhiều so với việc KHÔNG có nhãn cho 52,2% câu.
    """
    from collections import defaultdict
    idx_by_doc = defaultdict(list)
    for i, c in enumerate(all_chunks):
        idx_by_doc[str(c["id"]).split("_")[0]].append(i)

    with open(task1_path, encoding="utf-8") as f:
        t1 = json.load(f)
    pairs, extra_data, n_nodoc, n_dup = {}, {}, 0, 0
    for qid, item in t1.items():
        q = item.get("question")
        gold = item.get("answer") or []
        if not isinstance(q, str) or not gold:
            continue
        if q in seen_questions:          # 21 câu trùng với Task 2 -> bỏ, tránh lẫn vào dev
            n_dup += 1
            continue
        cand = [i for d in gold for i in idx_by_doc.get(str(d), [])]
        if not cand:
            n_nodoc += 1
            continue
        scores = bm25.get_scores(tokenize_simple(q))
        best = max(cand, key=lambda i: scores[i])
        key = f"task1_{qid}"
        pairs[key] = all_chunks[best]["id"]
        extra_data[key] = {"question": q, "answer": ""}
    print(f"  Task 1: {len(t1)} câu -> {len(pairs)} positive pair"
          + (f" · {n_dup} câu trùng Task 2 (bỏ)" if n_dup else "")
          + (f" · {n_nodoc} câu không tra được document (bỏ)" if n_nodoc else ""))
    return pairs, extra_data


n_task1_added = 0
if USE_TASK1_LABELS and os.path.exists(TASK1_TRAIN_PATH):
    _t1_pairs, _t1_data = build_task1_pairs(
        TASK1_TRAIN_PATH, all_chunks, bm25, {v["question"] for v in train_data.values()})
    # Nhãn citation ĐI TRƯỚC: nó ở mức Điều và chính xác hơn. Task 1 chỉ BỔ SUNG cho
    # những câu chưa có nhãn, không ghi đè — cùng nguyên tắc "bổ sung, không thay thế"
    # mà analysis.md §3 của Task 1 đã rút ra sau khi làm ngược lại và mất điểm.
    _before = len(train_positive)
    for k, v in _t1_pairs.items():
        train_positive.setdefault(k, v)
    train_data_for_pairs.update(_t1_data)
    chunk_by_id = {c["id"]: c for c in all_chunks}
    n_task1_added = len(train_positive) - _before
    print(f"  Positive pairs: {_before} (citation) + {n_task1_added} (Task 1) "
          f"= {len(train_positive)}")
elif USE_TASK1_LABELS:
    print(f"  [CẢNH BÁO] USE_TASK1_LABELS=True nhưng không thấy {TASK1_TRAIN_PATH} — "
          f"thêm train.json của Task 1 vào Kaggle Dataset. Tạm chạy bằng nhãn citation.")

checkpoint("Xong sinh nhãn")


# Cell 8: Bước 4 — Fine-tune 2 dense encoder SONG SONG THẬT trên 2 GPU riêng (subprocess)
#
# CHỦ Ý dùng subprocess (không phải threading/multiprocessing.Process kiểu fork): Cell 3 đã
# init CUDA context trong tiến trình notebook (gọi torch.cuda.get_device_properties) — fork
# SAU khi CUDA đã init là lỗi kinh điển ("Cannot re-initialize CUDA in forked subprocess").
# subprocess.Popen luôn khởi động tiến trình Python HOÀN TOÀN MỚI (tương đương spawn), mỗi
# tiến trình con tự import torch riêng, tự nhận CUDA_VISIBLE_DEVICES riêng — an toàn tuyệt
# đối, đúng pattern `run_shards` của bản gốc `run_qa.py` đầu dự án.
#
# Checkpoint ĐƯỢC LƯU lần này (khác các bản trước) — bạn cần dùng lại qua nhiều phiên Kaggle,
# và vì tiến trình con/cha là 2 process riêng, cách DUY NHẤT đưa model đã train về tiến trình
# cha là qua đĩa (`model.save_pretrained()` rồi `SentenceTransformer(path)` load lại).
import subprocess
import sys  # SỬA: sys.executable dùng để gọi WORKER_SCRIPT bên dưới — thiếu import này\n# là bug thật (Python vẫn cho phép dùng module chưa import NẾU nó tình cờ đã có trong\n# builtins/đã import ở cell khác cùng kernel session — dễ chạy "trót lọt" trong notebook\n# rồi lỗi khó hiểu khi chạy .py độc lập; luôn import tường minh module mình dùng).

print("=== Bước 4: Fine-tune 2 dense encoder song song (bge-m3 @ cuda:0, e5-large @ cuda:1) ===")

WORKER_SCRIPT = os.path.join(CACHE_DIR, "_train_encoder_worker.py")
# Worker con — fine-tune MOT SentenceTransformer tren MOT GPU, chay qua subprocess.Popen,
# nhan tham so qua argv, khong phu thuoc bien toan cuc cua notebook.
worker_code = '''
import argparse, json, os, sys, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--gpu-index", required=True)
    p.add_argument("--rows-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mini-batch-size", type=int, default=16)
    p.add_argument("--time-budget-sec", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--query-prefix", default="")
    p.add_argument("--passage-prefix", default="")
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--use-8bit-optim", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        from datasets import Dataset
        from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                            SentenceTransformerTrainingArguments)
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
        import sentence_transformers as _st
        import accelerate as _acc
    except ImportError as e:
        print(f"[LOI IMPORT] {e}", flush=True)
        print(f"[LOI IMPORT] Ban co the dang dung sentence-transformers/accelerate qua cu -- "
              f"CachedMultipleNegativesRankingLoss can sentence-transformers >= 3.0, "
              f"accelerate >= 1.1.0. Chay:", flush=True)
        print(f'    pip install -U "sentence-transformers>=3.0" "accelerate>=1.1.0"', flush=True)
        sys.exit(1)
    st_ver = tuple(int(x) for x in _st.__version__.split(".")[:2] if x.isdigit())
    if st_ver < (3, 0):
        print(f"[PHIEN BAN CU] sentence-transformers={_st.__version__} (can >= 3.0). Chay: "
              f'pip install -U "sentence-transformers>=3.0"', flush=True)
        sys.exit(1)

    # BAN SUA (log loi that: torch.AcceleratorError OOM ngay o optimizer.step() DAU TIEN --
    # AdamW full fine-tune cho model ~568M can ~6-7GB CHI RIENG optimizer state, khong phu
    # thuoc batch size -- khong batch nao du tren GPU 4GB). LoRA: chi train 1 phan rat nho
    # tham so (dong bang phan con lai) -> optimizer state nho lai theo dung ty le do, GIAI
    # QUYET DUOC loai OOM nay ma batch-backoff khong the giai quyet.
    optim_name = "adamw_torch"
    if args.use_8bit_optim:
        try:
            import bitsandbytes  # noqa: F401
            optim_name = "adamw_bnb_8bit"
            print(f"[{args.base_model}] Dung optimizer AdamW 8-bit (bitsandbytes).", flush=True)
        except ImportError:
            print(f"[{args.base_model}] bitsandbytes khong cai duoc -> dung AdamW thuong.", flush=True)

    with open(args.rows_path, encoding="utf-8") as f:
        rows = json.load(f)
    if args.query_prefix or args.passage_prefix:
        fixed = []
        for r in rows:
            r2 = dict(r)
            r2["anchor"] = args.query_prefix + r["anchor"]
            for k in r:
                if k.startswith("positive") or k.startswith("negative"):
                    r2[k] = args.passage_prefix + r[k]
            fixed.append(r2)
        rows = fixed
    dataset = Dataset.from_list(rows)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_seq_len

    lora_used = False
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                   lora_dropout=args.lora_dropout, bias="none",
                                   target_modules="all-linear")
            model[0].auto_model = get_peft_model(model[0].auto_model, lora_cfg)
            lora_used = True
            n_trainable = sum(pp.numel() for pp in model[0].auto_model.parameters() if pp.requires_grad)
            n_total = sum(pp.numel() for pp in model[0].auto_model.parameters())
            print(f"[{args.base_model}] LoRA bat: {n_trainable}/{n_total} tham so co the train "
                  f"({100*n_trainable/max(n_total,1):.2f}%)", flush=True)
        except Exception as e:
            print(f"[{args.base_model}] LoRA loi ({e}) -> full fine-tune (can nhieu VRAM hon).", flush=True)

    batch_size, mini_batch_size = args.batch_size, args.mini_batch_size
    max_steps, calib_time = 0, None
    t0 = time.time()
    for attempt in range(4):
        try:
            loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
            calib_steps = min(10, max(1, len(dataset) // batch_size))
            calib_args = SentenceTransformerTrainingArguments(
                output_dir=args.output_dir + "_tmp", max_steps=calib_steps,
                per_device_train_batch_size=batch_size, logging_steps=calib_steps + 1,
                save_strategy="no", report_to=[], disable_tqdm=True, fp16=(device == "cuda:0"),
                optim=optim_name)
            c0 = time.time()
            print(f"[{args.base_model}] calib training (batch={batch_size}, mini_batch={mini_batch_size})...", flush=True)
            SentenceTransformerTrainer(model=model, args=calib_args, train_dataset=dataset, loss=loss).train()
            calib_time = (time.time() - c0) / calib_steps

            budget_left = args.time_budget_sec - (time.time() - t0) - 60
            max_steps = max(0, int(budget_left / max(calib_time, 1e-6)))
            max_steps = min(max_steps, (len(dataset) // batch_size) * 8)
            print(f"[{args.base_model}] calib {calib_time:.2f}s/step, ngan sach con "
                  f"{budget_left/60:.1f} phut -> {max_steps} step", flush=True)

            if max_steps > 0:
                targs = SentenceTransformerTrainingArguments(
                    output_dir=args.output_dir + "_tmp", max_steps=max_steps,
                    per_device_train_batch_size=batch_size, learning_rate=2e-5,
                    warmup_steps=0.05, lr_scheduler_type="cosine",
                    logging_steps=max(1, max_steps // 20), save_strategy="no", report_to=[],
                    fp16=(device == "cuda:0"), optim=optim_name)
                SentenceTransformerTrainer(model=model, args=targs, train_dataset=dataset, loss=loss).train()
            break
        except Exception as e:
            if "out of memory" in str(e).lower() and mini_batch_size > 1:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass  # cache da can kiet toi muc khong con gi de don -- bo qua, cu lui batch
                mini_batch_size = max(1, mini_batch_size // 2)
                print(f"[{args.base_model}] OOM -> mini_batch_size={mini_batch_size}", flush=True)
                continue
            raise

    if lora_used:
        try:
            model[0].auto_model = model[0].auto_model.merge_and_unload()
            print(f"[{args.base_model}] Da merge LoRA vao model goc.", flush=True)
        except Exception as e:
            print(f"[{args.base_model}] Merge LoRA loi ({e}) -> luu adapter rieng.", flush=True)

    model.save_pretrained(args.output_dir)
    meta = {"max_steps": max_steps, "mini_batch_final": mini_batch_size,
            "calib_time_s": calib_time, "elapsed_s": time.time() - t0,
            "lora_used": lora_used, "optim": optim_name}
    with open(args.output_dir + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"[{args.base_model}] DONE -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
'''
with open(WORKER_SCRIPT, "w", encoding="utf-8") as f:
    f.write(worker_code)

# ---- Tạo training rows 1 LẦN trong tiến trình cha (dùng chung cho cả 2 encoder) ----
def _build_training_rows(train_positive, train_data, chunk_by_id, all_chunks, bm25, n_neg=N_NEG_PER_ROW):
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


def _build_doc_index(all_chunks):
    """doc_id -> list chunk_id CÙNG văn bản (dùng để mine hard-negative "sai Điều, đúng
    văn bản" cho reranker) — doc_id suy từ chunk_id dạng f"{doc_id}_{dieu}_{i}"."""
    idx = {}
    for c in all_chunks:
        doc_id = c["id"].rsplit("_", 2)[0]
        idx.setdefault(doc_id, []).append(c["id"])
    return idx


def _build_reranker_rows(train_positive, train_data, chunk_by_id, doc_index, all_chunks, bm25,
                          n_neg=N_NEG_PER_ROW):
    """Sinh training rows cho RERANKER — ưu tiên hard-negative CÙNG VĂN BẢN (sai Điều, đúng
    văn bản gold) thay vì BM25 top-60 chung chung như dense encoder dùng. Lý do (log thật):
    sau khi fine-tune reranker bằng negative kiểu BM25-top-60 cũ, Recall@1 SẬP (59,4%->44,8%)
    — nghi vấn negative quá dễ (thường khác hẳn văn bản), không dạy được việc phân biệt các
    Điều trong CÙNG 1 văn bản, đúng loại nhầm lẫn hay gặp nhất trong thực tế. Văn bản chỉ có
    1 Điều (không đủ sibling) thì lùi về BM25 top-60 như cũ để bù đủ n_neg."""
    rows = []
    n = len(train_positive)
    for i, (qid, pos_id) in enumerate(train_positive.items()):
        question = train_data[qid]["question"]
        pos_text = chunk_by_id[pos_id]["text"]
        doc_id = pos_id.rsplit("_", 2)[0]
        siblings = [cid for cid in doc_index.get(doc_id, []) if cid != pos_id]
        random.shuffle(siblings)
        neg_ids = siblings[:n_neg]
        if len(neg_ids) < n_neg:
            token_q = tokenize_simple(question)
            ranked = bm25.top_k(token_q, 60)
            fallback = [all_chunks[i2]["id"] for i2 in ranked[5:60]
                        if all_chunks[i2]["id"] != pos_id and all_chunks[i2]["id"] not in neg_ids]
            neg_ids += fallback[:n_neg - len(neg_ids)]
        if len(neg_ids) < n_neg:
            pool = [c["id"] for c in all_chunks if c["id"] != pos_id and c["id"] not in neg_ids]
            while len(neg_ids) < n_neg and pool:
                neg_ids.append(random.choice(pool))
        row = {"anchor": question, "positive": pos_text}
        for j, nid in enumerate(neg_ids[:n_neg]):
            row[f"negative_{j+1}"] = chunk_by_id[nid]["text"]
        rows.append(row)
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"    _build_reranker_rows (hard-neg cùng văn bản): {i+1}/{n}  "
                  f"({elapsed()/60:.1f} phút)")
    return rows


finetune_info = {"used_finetune": False, "reason": None, "n_pairs_available": len(train_positive),
                  "n_pairs_used": 0, "models": {}}
DENSE_CHANNELS = []  # điền ở cuối cell; embeddings điền ở Cell 9

# BẢN SỬA: build `rows` (anchor/positive/negative) MỘT LẦN, KHÔNG PHỤ THUỘC USE_FINETUNE của
# dense encoder — Cell 10 (fine-tune reranker) cần dùng lại đúng `rows` này. Trước đây rows chỉ
# được build bên trong nhánh "if use_finetune" của dense encoder, nên nếu USE_FINETUNE=False thì
# Cell 10 không có gì để fine-tune reranker dù USE_RERANKER_FINETUNE=True.
#
# BẢN SỬA THÊM: tách riêng `rows_clean` (CHỈ nhãn citation, chính xác tới từng Điều) khỏi
# `rows` (citation + Task 1) — nhãn Task 1 là GIÁM SÁT YẾU ở mức Điều (chọn Điều điểm BM25 cao
# nhất TRONG đúng văn bản gold — có thể sai Điều dù đúng văn bản, xem docstring
# build_task1_pairs() ở Cell 7). Dense encoder train bằng contrastive loss với nhiều negative,
# chịu nhiễu nhãn tốt — dùng cả 2 nguồn (`rows`) là hợp lý. Reranker train bằng margin ranking
# loss trên 1 cặp positive/negative mỗi lần, KHÔNG có gì làm mềm nhiễu — lỡ học "Điều sai nhưng
# đúng văn bản" thành positive thật sẽ kéo NGƯỢC độ chính xác, đúng kiểu lỗi khớp với quan sát
# thật: thêm nhãn Task 1 làm METEOR tăng mạnh (tìm đúng NHIỀU câu hỏi hơn) nhưng ROUGE-L gần
# như đứng yên (không chính xác hơn ở mức từng chữ) — nghi vấn hợp lý là do một phần nhãn Điều
# không hoàn toàn đúng đang lẫn vào huấn luyện. Reranker vì vậy CHỈ train trên `rows_clean`.
rows_needed = (USE_FINETUNE or USE_RERANKER_FINETUNE) and len(train_positive) >= MIN_TRAIN_PAIRS \
              and remaining() > 10 * 60
rows, rows_path = [], None
rows_clean = []
if rows_needed:
    train_positive_used = train_positive
    if len(train_positive) > MAX_TRAIN_EXAMPLES:
        sampled_qids = random.sample(list(train_positive.keys()), MAX_TRAIN_EXAMPLES)
        train_positive_used = {qid: train_positive[qid] for qid in sampled_qids}
        print(f"  Có {len(train_positive)} positive pairs, lấy mẫu {MAX_TRAIN_EXAMPLES} "
              f"(tái lập được nhờ SEED={SEED}).")
    finetune_info["n_pairs_used"] = len(train_positive_used)

    print(f"  Đang tạo training rows (dense encoder — citation + Task 1)...")
    rows = _build_training_rows(train_positive_used, train_data_for_pairs, chunk_by_id, all_chunks, bm25)
    rows_path = os.path.join(CACHE_DIR, "train_rows.json")
    with open(rows_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"  {len(rows)} rows -> {rows_path}")

    # Nhãn Task 1 được đánh dấu qua tiền tố khoá "task1_" ngay từ lúc tạo ở Cell 7
    # (`key = f"task1_{qid}"`) — lọc theo tiền tố này là đủ, không cần cấu trúc dữ liệu mới.
    clean_positive = {qid: cid for qid, cid in train_positive_used.items()
                       if not str(qid).startswith("task1_")}
    if clean_positive:
        print(f"  Đang tạo training rows (reranker — CHỈ citation, hard-negative CÙNG VĂN "
              f"BẢN, {len(clean_positive)} câu)...")
        _doc_index = _build_doc_index(all_chunks)
        rows_clean = _build_reranker_rows(clean_positive, train_data_for_pairs, chunk_by_id,
                                           _doc_index, all_chunks, bm25)
        print(f"  {len(rows_clean)} rows_clean (reranker)")
    else:
        print(f"  Không có nhãn citation nào trong mẫu train hiện tại -> reranker sẽ không "
              f"fine-tune được dù USE_RERANKER_FINETUNE=True (xem Cell 10).")

use_finetune = USE_FINETUNE and bool(rows)
if not use_finetune:
    reason = ("USE_FINETUNE=False" if not USE_FINETUNE else
               ("chưa có rows (xem lý do rows_needed=False ở trên)" if not rows else "?"))
    print(f"  {reason} -> dùng zero-shot cho cả 2 dense encoder, không fine-tune.")
    finetune_info["reason"] = reason
    from sentence_transformers import SentenceTransformer
    m_a = SentenceTransformer(BASE_DENSE_MODEL_A, device=DEVICES[0]); m_a.max_seq_length = DENSE_MAX_SEQ_LEN
    m_b = SentenceTransformer(BASE_DENSE_MODEL_B, device=DEVICES[-1]); m_b.max_seq_length = DENSE_MAX_SEQ_LEN
    DENSE_CHANNELS = [
        {"name": "bge-m3", "model": m_a, "embeddings": None, "query_prefix": "", "passage_prefix": ""},
        {"name": "e5-large", "model": m_b, "embeddings": None, "query_prefix": "query: ", "passage_prefix": "passage: "},
    ]
else:
    specs = [
        {"name": "bge-m3", "base_model": BASE_DENSE_MODEL_A, "gpu": DEVICES[0].split(":")[-1],
         "out": os.path.join(CHECKPOINT_DIR, "bge-m3-ft"), "query_prefix": "", "passage_prefix": ""},
        {"name": "e5-large", "base_model": BASE_DENSE_MODEL_B, "gpu": DEVICES[-1].split(":")[-1],
         "out": os.path.join(CHECKPOINT_DIR, "e5-large-ft"), "query_prefix": "query: ", "passage_prefix": "passage: "},
    ]

    # BẢN SỬA (research mới — tái dùng checkpoint): fingerprint gồm đúng những gì QUYẾT ĐỊNH
    # kết quả train (không gồm mini_batch/OOM-backoff — CachedMultipleNegativesRankingLoss làm
    # mini_batch KHÔNG đổi kết quả toán học, chỉ đổi tốc độ). Fingerprint LỆCH dù 1 chi tiết
    # -> train lại bình thường, không bao giờ tái dùng nhầm checkpoint không khớp cấu hình.
    def _dense_fingerprint(spec):
        return {"base_model": spec["base_model"], "n_pairs": finetune_info["n_pairs_used"],
                "seed": SEED, "use_lora": USE_LORA, "max_seq_len": DENSE_MAX_SEQ_LEN,
                "batch_size": TRAIN_BATCH_SIZE,
                "lora_r": LORA_R if USE_LORA else None, "lora_alpha": LORA_ALPHA if USE_LORA else None}

    def _checkpoint_reusable(spec):
        fp_path = spec["out"] + "_fingerprint.json"
        if not (REUSE_CHECKPOINT_IF_EXISTS and os.path.isdir(spec["out"]) and os.path.exists(fp_path)):
            return False
        try:
            with open(fp_path, encoding="utf-8") as f:
                old_fp = json.load(f)
            return old_fp == spec["fingerprint"]
        except Exception:
            return False

    for spec in specs:
        spec["fingerprint"] = _dense_fingerprint(spec)
        spec["reused"] = _checkpoint_reusable(spec)
        if spec["reused"]:
            print(f"  {spec['name']}: TÁI DÙNG checkpoint có sẵn ({spec['out']}) -- fingerprint "
                  f"khớp cấu hình hiện tại, bỏ qua fine-tune.")
    to_train = [sp for sp in specs if not sp["reused"]]

    # Chỉ chạy THẬT SỰ song song nếu >=2 spec CẦN train VÀ có >=2 GPU riêng biệt cho chúng —
    # nếu chỉ 1 GPU (hoặc chỉ 1 spec cần train), cả 2 subprocess tranh cùng 1 thẻ nếu phóng
    # cùng lúc (dễ OOM cả hai) -> chạy TUẦN TỰ.
    run_parallel = (len(to_train) > 1 and len(DEVICES) > 1
                     and to_train[0]["gpu"] != to_train[1]["gpu"])
    time_budget_each = max(600.0, min(remaining() - 5 * 60, FINETUNE_TIME_BUDGET_SEC)
                            / (1.0 if run_parallel else max(1, len(to_train))))
    if to_train:
        print(f"  Cần train: {[sp['name'] for sp in to_train]} — chạy "
              f"{'SONG SONG (2 GPU riêng)' if run_parallel else 'TUẦN TỰ'} "
              f"— ngân sách mỗi encoder ~{time_budget_each/60:.0f} phút.")
    else:
        print("  Cả 2 encoder đều tái dùng checkpoint — bỏ qua hoàn toàn bước fine-tune.")

    def _launch(spec):
        log_path = os.path.join(CACHE_DIR, f"train_{spec['name']}.log")
        cmd = [sys.executable, WORKER_SCRIPT,
               "--base-model", spec["base_model"], "--gpu-index", spec["gpu"],
               "--rows-path", rows_path, "--output-dir", spec["out"],
               "--max-seq-len", str(DENSE_MAX_SEQ_LEN), "--batch-size", str(TRAIN_BATCH_SIZE),
               "--mini-batch-size", str(TRAIN_MINI_BATCH_SIZE), "--time-budget-sec", str(time_budget_each),
               "--seed", str(SEED), "--query-prefix", spec["query_prefix"], "--passage-prefix", spec["passage_prefix"],
               "--lora-r", str(LORA_R), "--lora-alpha", str(LORA_ALPHA), "--lora-dropout", str(LORA_DROPOUT)]
        if USE_LORA:
            cmd.append("--use-lora")
        if USE_8BIT_OPTIM:
            cmd.append("--use-8bit-optim")
        lf = open(log_path, "w")
        print(f"  Khởi động fine-tune {spec['name']} trên GPU {spec['gpu']} -> log: {log_path}")
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT), lf

    failed = []
    if run_parallel:
        procs = [(spec, *_launch(spec)) for spec in to_train]
        print(f"  Đang chờ {len(procs)} tiến trình fine-tune song song...", flush=True)
        for spec, proc, lf in procs:
            rc = proc.wait()
            print(f"  {spec['name']}: xong, mã thoát {rc}")
            if rc != 0:
                failed.append(spec["name"])
            lf.close()
    else:
        for spec in to_train:
            proc, lf = _launch(spec)
            rc = proc.wait()
            print(f"  {spec['name']}: xong, mã thoát {rc}")
            if rc != 0:
                failed.append(spec["name"])
            lf.close()
    if failed:
        raise SystemExit(f"Fine-tune lỗi: {failed} — xem log trong {CACHE_DIR}/train_<tên>.log")

    from sentence_transformers import SentenceTransformer
    for spec in specs:
        meta_path = spec["out"] + "_meta.json"
        if spec["reused"]:
            with open(meta_path, encoding="utf-8") as f:
                m = json.load(f)
            m["reused"] = True
        else:
            with open(meta_path, encoding="utf-8") as f:
                m = json.load(f)
            # Lưu fingerprint SAU KHI train xong thành công -- lần chạy sau mới tái dùng được.
            with open(spec["out"] + "_fingerprint.json", "w", encoding="utf-8") as f:
                json.dump(spec["fingerprint"], f)
            m["reused"] = False
        finetune_info["models"][spec["name"]] = m
        print(f"  {spec['name']}: {'TÁI DÙNG' if spec['reused'] else str(m['max_steps']) + ' step'}"
              f"{'' if spec['reused'] else f', mini_batch cuối={m["mini_batch_final"]}, {m["elapsed_s"]/60:.1f} phút'}")

    m_a = SentenceTransformer(specs[0]["out"], device=DEVICES[0])
    m_b = SentenceTransformer(specs[1]["out"], device=DEVICES[-1])
    DENSE_CHANNELS = [
        {"name": "bge-m3", "model": m_a, "embeddings": None, "query_prefix": "", "passage_prefix": ""},
        {"name": "e5-large", "model": m_b, "embeddings": None, "query_prefix": "query: ", "passage_prefix": "passage: "},
    ]
    finetune_info["used_finetune"] = True
    print(f"  Checkpoint đã lưu trong {CHECKPOINT_DIR} — tự tải về nếu muốn dùng lại phiên sau.")

checkpoint("Xong Bước 4 (2 dense encoder)")


# Cell 9: Bước 5 — Encode toàn bộ corpus CHO CẢ 2 ENCODER (mỗi encoder dùng multi-process
# pool riêng, chạy TUẦN TỰ giữa 2 encoder — mỗi lần encode đã tận dụng CẢ 2 GPU, chạy đồng
# thời 2 pool cùng lúc sẽ tranh nhau GPU chứ không nhanh hơn)
print(f"=== Bước 5: Encode toàn bộ corpus cho {len(DENSE_CHANNELS)} encoder ===")
texts_raw = [c["text"] for c in all_chunks]

for ch in DENSE_CHANNELS:
    t0 = time.time()
    texts = [ch["passage_prefix"] + t for t in texts_raw] if ch["passage_prefix"] else texts_raw
    print(f"  [{ch['name']}] encode {len(texts)} chunk"
          + (f' (tiền tố "{ch["passage_prefix"]}")' if ch["passage_prefix"] else "") + " ...")
    model = ch["model"]
    if len(DEVICES) > 1 and DEVICES[0].startswith("cuda"):
        pool = model.start_multi_process_pool(target_devices=DEVICES)
        try:
            emb = model.encode_multi_process(texts, pool, batch_size=ENCODE_BATCH_SIZE,
                                              normalize_embeddings=True)
        finally:
            model.stop_multi_process_pool(pool)
    else:
        device = DEVICES[0]
        model = model.to(device)
        if device.startswith("cuda"):
            model = model.half()
        batch_size = ENCODE_BATCH_SIZE
        while True:
            try:
                emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                    show_progress_bar=True, normalize_embeddings=True, device=device)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    print(f"    [CUDA OOM] batch_size={batch_size} -> thử {batch_size // 2}")
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    continue
                raise
        ch["model"] = model
    ch["embeddings"] = emb
    print(f"    -> {emb.shape}, {time.time()-t0:.0f}s")

checkpoint("Xong encode corpus (2 encoder)")


# Cell 10: Bước 5b — Fine-tune reranker (nếu USE_RERANKER_FINETUNE, tái dùng `rows_clean`
# CHỈ nhãn citation của Bước 4 — KHÔNG dùng nhãn Task 1, xem giải thích ở Cell 8) RỒI tải
# 1 bản MỖI GPU để rerank song song thật ở Bước 6/7 (xem Cell 11)
#
# LÝ DO: kết quả dual-encoder thật (0.5215/0.4829) chỉ nhích rất ít so với single-encoder
# (0.5199/0.4806) dù retrieval mạnh hơn nhiều -> retrieval không còn là nút thắt chính,
# reranker ZERO-SHOT (AITeamVN/Vietnamese_Reranker, train trên Legal Zalo 2021 — KHÔNG phải
# đúng format "Điều X" của bài này) nhiều khả năng đang là trần chặn điểm tiếp theo.
#
# Huấn luyện bằng vòng lặp PyTorch thuần (KHÔNG dùng CrossEncoderTrainer của
# sentence-transformers) — tránh phụ thuộc API cross-encoder mới có thể không có ở mọi
# phiên bản cài qua Cell 1; margin ranking loss (điểm(positive) phải lớn hơn điểm(negative)
# ít nhất RERANKER_FT_MARGIN) — không cần thang điểm chuẩn hoá, chỉ cần đúng THỨ TỰ, ổn
# định hơn BCE/MSE cho một đầu hồi quy logit thô như model này.
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_reranker_on(device: str, source: str):
    for attempt in range(2):
        try:
            print(f"  Đang tải reranker {source} lên {device}"
                  f"{' — thử lại lần 2' if attempt else ''}...")
            tok = AutoTokenizer.from_pretrained(source)
            mdl = AutoModelForSequenceClassification.from_pretrained(source)
            mdl = mdl.to(device)
            if device.startswith("cuda"):
                mdl = mdl.half()
            mdl.eval()
            return mdl, tok
        except Exception as e:
            if attempt == 0:
                print(f"  [Lần 1 lỗi: {e}] thử lại sau 5s...")
                time.sleep(5)
                continue
            print(f"  [CẢNH BÁO] Không tải được reranker trên {device} ({e}) -> bỏ qua "
                  f"reranker trên thẻ này.")
            return None, None


# finetune_reranker: trả về (model đã .eval(), tokenizer, meta dict). KHÔNG raise nếu OOM —
# tự giảm batch_size và thử lại, giống pattern OOM-backoff đã dùng cho dense encoder/rerank.
#
# BẢN SỬA (log lỗi thật ở Bước 4: torch.AcceleratorError OOM ngay optimizer.step() ĐẦU TIÊN —
# AdamW full fine-tune cho model ~568M cần ~6-7GB CHỈ RIÊNG optimizer state, KHÔNG phụ thuộc
# batch_size — cùng lỗi này áp dụng y hệt cho reranker, cùng kích cỡ tham số). Khi
# USE_LORA=True: đóng băng gần hết tham số gốc, chỉ train adapter nhỏ — optimizer state co
# lại theo đúng tỷ lệ, giải quyết được loại OOM mà batch-backoff không giải quyết nổi. Khi
# USE_8BIT_OPTIM=True (+ bitsandbytes cài được): dùng AdamW 8-bit, giảm thêm ~4 lần bộ nhớ
# optimizer, cộng dồn với LoRA. torch.cuda.empty_cache() được bọc try/except riêng — khi VRAM
# cạn quá sâu, chính lệnh dọn cache cũng có thể OOM (đã gặp thật), không để nó làm sập
# toàn bộ vòng lặp backoff.
def finetune_reranker(rows, base_model: str, device: str, time_budget_sec: float,
                       batch_size: int, lr: float, margin: float, seed: int,
                       use_lora: bool = False, use_8bit_optim: bool = False,
                       lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    # `margin` giữ trong chữ ký để không phá lời gọi cũ, nhưng KHÔNG còn dùng trong loss —
    # đã đổi sang softplus(-(pos-neg)) (logistic pairwise, không cần margin) — xem lý do ở
    # bản sửa trong vòng lặp train bên dưới.
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model).to(device)

    lora_used = False
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                   bias="none", target_modules="all-linear",
                                   task_type=TaskType.SEQ_CLS)
            model = get_peft_model(model, lora_cfg)
            lora_used = True
            n_trainable = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
            n_total = sum(pp.numel() for pp in model.parameters())
            print(f"  [reranker-ft] LoRA bật: {n_trainable}/{n_total} tham số có thể train "
                  f"({100*n_trainable/max(n_total,1):.2f}%)")
        except Exception as e:
            print(f"  [reranker-ft] LoRA lỗi ({e}) -> full fine-tune (cần nhiều VRAM hơn).")
    model.train()

    if use_8bit_optim:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
            print("  [reranker-ft] Dùng optimizer AdamW 8-bit (bitsandbytes).")
        except ImportError:
            print("  [reranker-ft] bitsandbytes không cài được -> dùng AdamW thường.")
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    # BẢN SỬA (log lỗi thật: dev-eval đo được Recall@1 SẬP từ 59,4% (không rerank) xuống
    # 36,4% (có reranker fine-tune) — reranker fine-tune không chỉ kém, mà đang PHÁ HỎNG kết
    # quả. Log training cho thấy loss=0.0000 ở gần như MỌI bước suốt 5000+ step: margin
    # ranking loss (điểm(positive) phải hơn điểm(negative) ÍT NHẤT `margin`) bị BÃO HOÀ quá
    # sớm — reranker gốc đã pretrain tốt trên Legal Zalo 2021 nên chỉ cần ~100 bước đã thoả
    # mãn margin cho hầu hết cặp, hầu như không còn gradient cho ~5000 bước còn lại, chỉ lặp
    # đi lặp lại trên một nhóm nhỏ ví dụ khó/nhiễu còn sót -> overfit lệch. Đổi sang loss
    # logistic từng cặp (RankNet-style, softplus(-(pos-neg))) — KHÔNG BAO GIỜ về đúng 0, luôn
    # còn gradient tỉ lệ với độ tin cậy hiện tại, không có "margin" để bão hoà tới. Thêm dừng
    # sớm dựa trên loss trung bình cửa sổ trượt — nếu đã hội tụ thấp bền vững, dừng thay vì
    # tiếp tục ép học trên phần dư nhiễu tới hết ngân sách thời gian.
    EARLY_STOP_WINDOW, EARLY_STOP_THRESHOLD, EARLY_STOP_MIN_STEPS = 200, 0.05, 500
    loss_history = []
    early_stopped = False

    g = random.Random(seed)  # RNG RIÊNG — không đụng vào random toàn cục (đã seed cho Bước 3/4)
    order = list(range(len(rows)))
    bs = batch_size
    t0 = time.time()
    step, n_neg = 0, N_NEG_PER_ROW
    while time.time() - t0 < time_budget_sec and not early_stopped:
        g.shuffle(order)
        for i in range(0, len(order), bs):
            batch_idx = order[i:i + bs]
            if not batch_idx:
                continue
            batch_rows = [rows[j] for j in batch_idx]
            pos_pairs = [[r["anchor"], r["positive"]] for r in batch_rows]
            neg_pairs = [[r["anchor"], r[f"negative_{k+1}"]] for r in batch_rows for k in range(n_neg)]
            try:
                pos_in = tok(pos_pairs, padding=True, truncation=True, max_length=512,
                             return_tensors="pt").to(device)
                neg_in = tok(neg_pairs, padding=True, truncation=True, max_length=512,
                             return_tensors="pt").to(device)
                with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                    pos_scores = model(**pos_in).logits.view(-1)
                    neg_scores = model(**neg_in).logits.view(len(batch_rows), n_neg)
                    pos_exp = pos_scores.unsqueeze(1).expand_as(neg_scores)
                    loss = torch.nn.functional.softplus(-(pos_exp - neg_scores)).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                step += 1
            except Exception as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass  # cache đã cạn kiệt tới mức không còn gì để dọn -- bỏ qua, cứ lùi batch
                    bs = max(1, bs // 2)
                    print(f"  [CUDA OOM reranker-ft] batch_size -> {bs}")
                    continue
                raise
            loss_history.append(loss.item())
            if len(loss_history) > EARLY_STOP_WINDOW:
                loss_history.pop(0)
            if (step >= EARLY_STOP_MIN_STEPS and len(loss_history) == EARLY_STOP_WINDOW
                    and sum(loss_history) / EARLY_STOP_WINDOW < EARLY_STOP_THRESHOLD):
                print(f"    reranker-ft DỪNG SỚM tại step {step}: loss trung bình "
                      f"{EARLY_STOP_WINDOW} bước gần nhất = "
                      f"{sum(loss_history)/EARLY_STOP_WINDOW:.4f} < {EARLY_STOP_THRESHOLD} "
                      f"(đã hội tụ, tránh overfit lên phần dư nhiễu)", flush=True)
                early_stopped = True
                break
            if time.time() - t0 >= time_budget_sec:
                break
            if step % 100 == 0:
                print(f"    reranker-ft step {step}, loss={loss.item():.4f}, "
                      f"{(time.time()-t0)/60:.1f} phút", flush=True)

    if lora_used:
        try:
            model = model.merge_and_unload()
            print("  [reranker-ft] Đã merge LoRA vào model gốc.")
        except Exception as e:
            print(f"  [reranker-ft] Merge LoRA lỗi ({e}) -> lưu nguyên adapter.")

    model.eval()
    meta = {"steps": step, "batch_size_final": bs, "elapsed_s": time.time() - t0,
            "lora_used": lora_used, "early_stopped": early_stopped}
    return model, tok, meta


print("=== Bước 5b: Fine-tune reranker (nếu bật) + tải mỗi GPU 1 bản ===")
reranker_finetune_info = {"used": False, "reason": None, "steps": 0, "elapsed_s": 0.0}
reranker_source = RERANKER_BASE
use_reranker_finetune = USE_RERANKER_FINETUNE and bool(rows_clean) and remaining() > 15 * 60
reranker_ckpt = os.path.join(CHECKPOINT_DIR, "reranker-ft")

# BẢN SỬA (research mới — tái dùng checkpoint): cùng nguyên tắc ở Bước 4 — fingerprint gồm
# đúng những gì quyết định kết quả train, lệch dù 1 chi tiết thì train lại bình thường.
_rr_fingerprint = {"base_model": RERANKER_BASE, "n_pairs": len(rows_clean), "seed": SEED,
                    "use_lora": USE_LORA, "lora_r": LORA_R if USE_LORA else None,
                    "lora_alpha": LORA_ALPHA if USE_LORA else None}
_rr_fp_path = reranker_ckpt + "_fingerprint.json"
_rr_reused = False
if use_reranker_finetune and REUSE_CHECKPOINT_IF_EXISTS and os.path.isdir(reranker_ckpt) and os.path.exists(_rr_fp_path):
    try:
        with open(_rr_fp_path, encoding="utf-8") as f:
            _old_fp = json.load(f)
        _rr_reused = (_old_fp == _rr_fingerprint)
    except Exception:
        _rr_reused = False

if not use_reranker_finetune:
    reason = ("USE_RERANKER_FINETUNE=False" if not USE_RERANKER_FINETUNE else
              ("không có rows_clean (xem Bước 4 — có thể toàn bộ nhãn hiện tại đến từ Task 1)"
               if not rows_clean else "hết ngân sách thời gian"))
    print(f"  {reason} -> reranker giữ ZERO-SHOT ({RERANKER_BASE}).")
    reranker_finetune_info["reason"] = reason
elif _rr_reused:
    print(f"  TÁI DÙNG checkpoint reranker có sẵn ({reranker_ckpt}) -- fingerprint khớp cấu "
          f"hình hiện tại, bỏ qua fine-tune.")
    reranker_source = reranker_ckpt
    meta_path = reranker_ckpt + "_meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    reranker_finetune_info.update({"used": True, "reused": True, **meta})
else:
    t0 = time.time()
    ft_model, ft_tok, meta = finetune_reranker(
        rows_clean, RERANKER_BASE, DEVICES[0], RERANKER_FT_TIME_BUDGET_SEC,
        RERANKER_FT_BATCH_SIZE, RERANKER_FT_LR, RERANKER_FT_MARGIN, SEED,
        use_lora=USE_LORA, use_8bit_optim=USE_8BIT_OPTIM,
        lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
    ft_model.half().save_pretrained(reranker_ckpt)  # lưu fp16 — nhất quán với cách nạp lại để rerank
    ft_tok.save_pretrained(reranker_ckpt)
    del ft_model
    torch.cuda.empty_cache()
    reranker_source = reranker_ckpt
    reranker_finetune_info.update({"used": True, "reused": False, **meta})
    with open(reranker_ckpt + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(_rr_fp_path, "w", encoding="utf-8") as f:
        json.dump(_rr_fingerprint, f)
    print(f"  Fine-tune reranker xong: {meta['steps']} step, {meta['elapsed_s']/60:.1f} phút "
          f"-> checkpoint {reranker_ckpt}")

# BẢN SỬA (log lỗi thật: dev-eval đo được reranker fine-tune làm Recall@1 SẬP 59,4% -> 36,4%
# — fine-tune "thành công" về mặt kỹ thuật nhưng model tệ hơn hẳn zero-shot). LƯỚI AN TOÀN:
# LUÔN tải zero-shot làm baseline; nếu có fine-tune, tải THÊM bản fine-tune riêng — Bước 6
# (Cell 12) sẽ so sánh CẢ HAI bằng dev-eval thật rồi mới quyết định dùng bản nào cho Bước 7,
# thay vì mặc định tin fine-tune luôn tốt hơn. Đảm bảo không bao giờ tệ hơn baseline zero-shot
# cũ nữa, bất kể fine-tune có thật sự cải thiện hay không lần chạy đó.
reranker_models, reranker_tokenizers = {}, {}          # ZERO-SHOT — luôn tải, dùng làm baseline
for dev in DEVICES:
    m, t = load_reranker_on(dev, RERANKER_BASE)
    if m is not None:
        reranker_models[dev] = m
        reranker_tokenizers[dev] = t

reranker_models_ft, reranker_tokenizers_ft = {}, {}     # FINE-TUNED — chỉ tải thêm nếu có fine-tune
if reranker_finetune_info["used"]:
    for dev in DEVICES:
        m, t = load_reranker_on(dev, reranker_source)
        if m is not None:
            reranker_models_ft[dev] = m
            reranker_tokenizers_ft[dev] = t

HAS_RERANKER = len(reranker_models) > 0
HAS_RERANKER_FT = len(reranker_models_ft) > 0
RERANK_DEVICES = list(reranker_models.keys())
print(f"  Reranker zero-shot sẵn sàng trên: {RERANK_DEVICES or '(không tải được)'}"
      + (f" | fine-tuned CŨNG sẵn sàng trên: {list(reranker_models_ft.keys())} "
         f"(Bước 6 sẽ so sánh, tự chọn cái thắng)" if HAS_RERANKER_FT else ""))
checkpoint("Xong tải reranker")


# Cell 11: Hàm retrieval (RRF fusion N kênh — BM25 + N encoder) + rerank theo lô
# + hạ tầng chạy song song 2 GPU
from concurrent.futures import ThreadPoolExecutor

_print_lock = __import__("threading").Lock()

def rrf_retrieve(question: str, bm25, dense_channels, all_chunks, top_k: int = TOP_K_RETRIEVE,
                  return_scores: bool = False):
    """RRF fusion N kênh: BM25 + mỗi encoder trong `dense_channels` (list of {"model",
    "embeddings", "query_prefix"}). Mỗi encoder có thể cần tiền tố khác nhau lúc encode QUERY
    (vd e5: "query: ") — PHẢI khớp tiền tố đã dùng lúc encode CORPUS ở Cell 9, nếu không
    embedding lệch hệ toạ độ mà không lỗi nào báo (bẫy đã ghi ở Cell 2/8).

    BẢN SỬA (log thật: Recall@1 KHÔNG rerank = 60,1%, rerank zero-shot = 55,2%, rerank
    fine-tuned = 44,8% — rerank làm Recall@k TỆ ĐI ở MỌI mức k, dù hệ thống vẫn chọn dùng
    rerank vì adaptive-k trên điểm reranker cho METEOR cuối cao hơn top_n cố định không
    rerank). Gợi ý: giá trị thật nằm ở "adaptive-k" (biết khi nào dừng), không nằm ở bản
    thân việc rerank. `return_scores=True` trả thêm mảng điểm RRF (đã sort giảm dần, khớp
    thứ tự chunk trả về) để thử `adaptive_k_cutoff` TRỰC TIẾP trên điểm RRF — không cần
    rerank — xem cấu hình mới ở Cell 12."""
    bm25_ranked = bm25.top_k(tokenize_simple(question), top_k)
    rank_maps = [{idx: r for r, idx in enumerate(bm25_ranked)}]
    all_idx = set(bm25_ranked)
    for ch in dense_channels:
        q_text = ch["query_prefix"] + question if ch["query_prefix"] else question
        q_emb = ch["model"].encode([q_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = ch["embeddings"] @ q_emb
        ranked_ch = list(np.argsort(-scores)[:top_k])
        rank_maps.append({idx: r for r, idx in enumerate(ranked_ch)})
        all_idx |= set(ranked_ch)
    rrf = {i: sum(1 / (60 + rm.get(i, top_k + 1)) for rm in rank_maps) for i in all_idx}
    ranked = sorted(rrf, key=rrf.get, reverse=True)
    chunks = [all_chunks[i] for i in ranked]
    if return_scores:
        rrf_scores = np.array([rrf[i] for i in ranked], dtype=np.float32)
        return chunks, rrf_scores
    return chunks


def rerank(question: str, candidates: list, reranker_model, reranker_tokenizer,
           max_candidates: int = TOP_K_RETRIEVE, max_length: int = 1024, sub_batch: int = RERANK_SUBBATCH):
    """Chấm điểm lại top `max_candidates` bằng cross-encoder, xử lý theo LÔ NHỎ (`sub_batch`)
    thay vì nhồi hết `max_candidates` vào 1 forward — kết quả điểm số KHÔNG đổi (transformer
    không trộn phép tính giữa các example trong batch, chỉ có padding — lô nhỏ padding ít
    hơn), chỉ nhanh hơn và ổn định VRAM hơn. Trả thêm điểm số cho adaptive_k_cutoff()."""
    if reranker_model is None or not candidates:
        return candidates, None
    subset = candidates[:max_candidates]
    device = next(reranker_model.parameters()).device
    pairs = [[question, c["text"]] for c in subset]
    scores = np.empty(len(pairs), dtype=np.float32)
    bs, i = max(1, sub_batch), 0
    while i < len(pairs):
        batch = pairs[i:i + bs]
        try:
            with torch.no_grad():
                inputs = reranker_tokenizer(batch, padding=True, truncation=True,
                                             return_tensors="pt", max_length=max_length).to(device)
                out = reranker_model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()
            scores[i:i + len(batch)] = out
            i += bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                continue
            if "out of memory" in str(e).lower():
                return candidates, None
            raise
    order = np.argsort(-scores)
    reranked = [subset[i2] for i2 in order]
    sorted_scores = scores[order]
    return reranked + candidates[max_candidates:], sorted_scores


def adaptive_k_cutoff(scores, min_k: int = 1, max_k: int = TOP_K_RERANK, search_window: int = 15) -> int:
    """Adaptive-k (Taguchi et al. 2025, arXiv:2506.08479): tìm điểm "gãy" tự nhiên trong
    phân phối điểm reranker đã sort giảm dần thay vì luôn cắt ở top_n cố định."""
    if scores is None or len(scores) == 0:
        return min_k
    n = min(len(scores), search_window)
    if n <= 1:
        return min_k
    gaps = [scores[i] - scores[i + 1] for i in range(n - 1)]
    k_star = int(np.argmax(gaps)) + 1
    return max(min_k, min(k_star, max_k))


def render_answer(selected_chunks: list, top_n: int, question: str = "",
                   concl: str = CONCL) -> str:
    """Câu dẫn "Căn cứ Điều X <loại VB> <số hiệu> quy định như sau:" — khuôn phổ biến nhất
    đo được trên answer thật (57.4% mở đầu "Căn cứ", 24.6% có "quy định như sau"). Cắt bỏ
    "Điều X." lặp lại ở đầu thân bài (98.8% answer thật không lặp).

    CÂU KẾT (`concl`) — thay đổi ĐÁNG GIÁ NHẤT và rẻ nhất trong cả pipeline. Đo trên
    501 câu dev, cùng retrieval, chỉ đổi một biến:

        concl=none    METEOR 0,5151
        concl=echo    METEOR 0,5499    Δ +0,0348 ± 0,0017 · 416 thắng / 85 thua · t = 20,4
        concl=echo2   METEOR 0,5630    Δ +0,0131 ± 0,0010 · 371 thắng / 130 thua · t = 12,8

    Split-half (chia đôi dev, chọn trên nửa này đo nửa kia): CẢ HAI nửa độc lập đều chọn
    echo2 (A 0,5675 · B 0,5589) — nên đây không phải ảo giác đỉnh-trên-toàn-dev.

    Vì sao ăn điểm: METEOR có alpha = 0,9 nên nặng recall, và 36,2% đáp án thật chứa
    "Như vậy", 27,5% chứa "Theo đó" — chúng nhắc lại nội dung câu hỏi ở phần kết. Lặp
    lại câu hỏi làm khớp đúng nhóm token đó.

    Đã dò tiếp số lần lặp: 1× 0,5499 · 2× 0,5630 · 3× 0,5674 · 4× 0,5682 · 6× 0,5661.
    Có đỉnh thật quanh 4, NHƯNG dừng ở 2: từ 2 lên 4 chỉ được +0,5 điểm (đúng vùng mà
    "chọn đỉnh trên toàn dev" đã lừa project này ba lần), còn đáp án lặp câu hỏi bốn lần
    thì nhìn bằng mắt là hỏng rõ ràng. Đây là tối ưu HÌNH DẠNG ĐỘ ĐO, hợp lệ theo luật
    nhưng không làm câu trả lời tốt hơn cho người đọc — biết để không đi xa hơn một cách
    mù quáng."""
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
    ans = "\n\n".join(parts)
    if concl != "none" and question:
        q = question.strip().rstrip("?").strip()
        if q:
            ql = q[0].lower() + q[1:]
            if concl == "echo":
                ans += f"\nNhư vậy, theo quy định nêu trên thì {ql}."
            elif concl == "echo2":
                ans += f"\nTheo đó, {ql}.\nNhư vậy, theo quy định nêu trên thì {ql}."
    return ans


def answer_question(question: str, bm25, dense_channels, all_chunks, top_n: int,
                     reranker_model=None, reranker_tokenizer=None, use_adaptive_k: bool = False,
                     use_rrf_gaps: bool = False) -> str:
    """use_rrf_gaps=True: adaptive-k chạy TRỰC TIẾP trên điểm RRF fusion, KHÔNG rerank —
    thêm sau khi đo được rerank làm Recall@k tệ đi ở MỌI mức k (log thật, xem Cell 12) dù
    hệ thống vẫn có lợi từ adaptive-k; giả thuyết: lợi ích nằm ở CƠ CHẾ adaptive-k, không
    nằm ở bản thân việc rerank — thử adaptive-k ngay trên nguồn có Recall cao nhất (RRF)."""
    if use_rrf_gaps:
        ranked, scores = rrf_retrieve(question, bm25, dense_channels, all_chunks, return_scores=True)
        if not ranked:
            return "Không tìm thấy thông tin pháp lý cho câu hỏi này."
        n = adaptive_k_cutoff(scores) if use_adaptive_k else top_n
        return render_answer(ranked, n, question)
    ranked = rrf_retrieve(question, bm25, dense_channels, all_chunks)
    if not ranked:
        return "Không tìm thấy thông tin pháp lý cho câu hỏi này."
    scores = None
    if reranker_model is not None:
        ranked, scores = rerank(question, ranked, reranker_model, reranker_tokenizer)
    n = adaptive_k_cutoff(scores) if (use_adaptive_k and scores is not None) else top_n
    return render_answer(ranked, n, question)


def split_evenly(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def parallel_process(ids, worker_fn, devices, label: str = "", progress_every: int = 50):
    """Chia `ids` đều cho từng thiết bị trong `devices`, chạy worker_fn(ids_chunk, device)
    ĐỒNG THỜI trên các luồng riêng. PyTorch giải phóng GIL trong lúc chờ CUDA hoàn thành nên
    2 luồng ghim vào 2 GPU vật lý khác nhau chạy song song THẬT (không phải giả song song do
    GIL) — đây là chỗ mang lại tốc độ x~2 cho phần rerank ở Bước 6/7.
    worker_fn(ids_chunk, device, worker_idx) -> dict {qid: kết quả}."""
    ids = list(ids)
    devices = list(devices) if devices else ["cpu"]
    chunks = split_evenly(ids, len(devices))
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futures = [ex.submit(worker_fn, chunk, dev, i) for i, (chunk, dev) in enumerate(zip(chunks, devices))]
        for f in futures:
            results.update(f.result())
    print(f"    [{label}] {len(ids)} câu / {len(devices)} thiết bị song song -> {time.time()-t0:.0f}s")
    return results


def _progress_print(label, worker_idx, i, n):
    if (i + 1) % 50 == 0 or (i + 1) == n:
        with _print_lock:
            print(f"    [{label} · luồng {worker_idx}] {i+1}/{n}")

# Cell 12: Bước 6 — Dev-eval (chọn TOP_N_ANSWER, dùng reranker nào — không/zero-shot/
# fine-tuned — hay adaptive-k) ĐỒNG THỜI đo Recall@k — gộp chung 1 lượt retrieval+rerank.
#
# BẢN SỬA (log lỗi thật: reranker fine-tune làm Recall@1 SẬP 59,4%->36,4%, dev-eval trước
# đây chỉ so "không rerank" vs "rerank" — không có lựa chọn "zero-shot" nếu đã fine-tune,
# nên khi fine-tune hỏng thì MẤT LUÔN lợi ích của zero-shot, không chỉ mất phần fine-tune):
# giờ so sánh CẢ BA — không rerank / rerank zero-shot / rerank fine-tuned (nếu có) — bằng
# METEOR đo thật trên cùng 300 câu dev, chọn đúng cái thắng. Đảm bảo không bao giờ tệ hơn
# baseline zero-shot cũ nữa.
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

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
# BẢN SỬA: re-seed CỐ Ý ở đây (dùng SEED chung, không phải số 42 rời rạc) — đảm bảo mẫu
# dev-eval LUÔN CỐ ĐỊNH bất kể USE_WARMUP/USE_FINETUNE bật hay tắt (số lời gọi random.*
# trước đó thay đổi tuỳ cấu hình sẽ làm lệch state nếu không re-seed ở đây) — cần thiết để
# ablation so sánh đúng nghĩa "cùng 300 câu, chỉ khác 1 biến". Xem PHAN_TICH_KY_THUAT.md §5.2.
random.seed(SEED)

print("=== Bước 6: Dev-eval chọn TOP_N_ANSWER + đo Recall@k ===")
n_sample = min(DEV_EVAL_SAMPLE_SIZE, len(train_data))
dev_ids = random.sample(list(train_data.keys()), n_sample)
recall_ids = [q for q in dev_ids if q in train_positive]
print(f"  Mẫu dev-eval: {len(dev_ids)} câu ({len(recall_ids)} câu có citation resolve được "
      f"-> dùng luôn để đo Recall@k, không chạy lại retrieval riêng).")

ks = [1, 3, 5, 10, 30, 100]
configs = [("BM25+dense (không rerank)", False, None)]
if HAS_RERANKER:
    configs.append(("BM25+dense+rerank (zero-shot)", True, "zeroshot"))
if HAS_RERANKER_FT:
    configs.append(("BM25+dense+rerank (fine-tuned)", True, "finetuned"))

# BẢN SỬA: theo dõi thêm best_r (ROUGE-L của cấu hình thắng) + recall_at_k_by_label — để Cell 14
# ghi đủ vào sổ thí nghiệm (trước đây các con số này chỉ in ra console rồi mất).
best_n, best_m, best_r, best_use_rerank, best_use_adaptive = 3, -1.0, None, False, False
best_rr_source = None   # None | "zeroshot" | "finetuned" — CÁI THẮNG, quyết định reranker dùng ở Bước 7
best_use_rrf_gaps = False  # True nếu cấu hình RRF-gaps (không rerank) thắng dev-eval
recall_at_k_by_label = {}
for label, use_rr, rr_source in configs:
    print(f"  --- {label} ---")
    rr_models = reranker_models_ft if rr_source == "finetuned" else reranker_models
    rr_tokenizers = reranker_tokenizers_ft if rr_source == "finetuned" else reranker_tokenizers
    rr_devices = list(rr_models.keys())
    if use_rr and len(rr_devices) > 1:
        def _worker(chunk, dev, widx, _label=label, _rr_models=rr_models, _rr_tok=rr_tokenizers):
            out = {}
            for i, qid in enumerate(chunk):
                item = train_data[qid]
                ranked = rrf_retrieve(item["question"], bm25, DENSE_CHANNELS, all_chunks)
                scores = None
                if ranked:
                    ranked, scores = rerank(item["question"], ranked,
                                             _rr_models[dev], _rr_tok[dev])
                out[qid] = (ranked, scores)
                _progress_print(_label, widx, i, len(chunk))
            return out
        merged = parallel_process(dev_ids, _worker, rr_devices, label=label)
        ranked_cache = {q: v[0] for q, v in merged.items()}
        scores_cache = {q: v[1] for q, v in merged.items()}
    else:
        # Không rerank (rẻ, tuần tự đủ nhanh) hoặc chỉ 1 GPU cho reranker.
        rr_dev = rr_devices[0] if (use_rr and rr_devices) else None
        ranked_cache, scores_cache = {}, {}
        t0 = time.time()
        for i, qid in enumerate(dev_ids):
            item = train_data[qid]
            ranked = rrf_retrieve(item["question"], bm25, DENSE_CHANNELS, all_chunks)
            scores = None
            if rr_dev is not None and ranked:
                ranked, scores = rerank(item["question"], ranked,
                                         rr_models[rr_dev], rr_tokenizers[rr_dev])
            ranked_cache[qid] = ranked
            scores_cache[qid] = scores
            if (i + 1) % 50 == 0 or (i + 1) == len(dev_ids):
                print(f"    retrieval+rerank {i+1}/{len(dev_ids)} ... {time.time()-t0:.0f}s")

    if recall_ids:
        hits = {k: 0 for k in ks}
        for qid in recall_ids:
            ranked_ids = [c["id"] for c in ranked_cache[qid]]
            pos_id = train_positive[qid]
            for k in ks:
                if pos_id in ranked_ids[:k]:
                    hits[k] += 1
        nr = len(recall_ids)
        print(f"  Recall@k ({label}, n={nr}):")
        for k in ks:
            print(f"    Recall@{k:<3d} = {hits[k]}/{nr} = {100*hits[k]/nr:.1f}%")
        recall_at_k_by_label[label] = {str(k): round(hits[k] / nr, 4) for k in ks}

    for top_n in range(1, TOP_K_RERANK + 1):
        ms, rs = [], []
        for qid in dev_ids:
            ranked = ranked_cache[qid]
            pred = render_answer(ranked, top_n, train_data[qid]["question"]) if ranked else "Không tìm thấy thông tin pháp lý cho câu hỏi này."
            ref = train_data[qid]["answer"]
            ms.append(meteor_score([str(ref).split()], str(pred).split()))
            rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
        m, r = sum(ms) / len(ms), sum(rs) / len(rs)
        print(f"    top_n={top_n}  METEOR={m:.4f}  ROUGE-L={r:.4f}  (n={len(dev_ids)})")
        if m > best_m:
            best_m, best_r, best_n = m, r, top_n
            best_use_rerank, best_use_adaptive, best_rr_source = use_rr, False, rr_source

    if use_rr:
        ms, rs = [], []
        for qid in dev_ids:
            ranked, scores = ranked_cache[qid], scores_cache[qid]
            k = adaptive_k_cutoff(scores) if ranked else 1
            pred = render_answer(ranked, k, train_data[qid]["question"]) if ranked else "Không tìm thấy thông tin pháp lý cho câu hỏi này."
            ref = train_data[qid]["answer"]
            ms.append(meteor_score([str(ref).split()], str(pred).split()))
            rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
        m, r = sum(ms) / len(ms), sum(rs) / len(rs)
        print(f"    adaptive-k       METEOR={m:.4f}  ROUGE-L={r:.4f}  (n={len(dev_ids)})")
        if m > best_m:
            best_m, best_r, best_use_rerank, best_use_adaptive = m, r, True, True
            best_rr_source = rr_source

# BẢN SỬA — thử adaptive-k TRỰC TIẾP trên điểm RRF fusion, KHÔNG rerank (xem docstring
# answer_question() ở Cell 11): log thật cho thấy rerank làm Recall@k tệ đi ở MỌI mức k so
# với RRF thuần, nhưng hệ thống vẫn có lợi từ CƠ CHẾ adaptive-k — thử adaptive-k ngay trên
# nguồn có Recall cao nhất, không cần rerank, không tốn thêm GPU nào.
print("  --- BM25+dense (RRF gaps, không rerank) ---")
rrf_ranked_cache, rrf_scores_cache = {}, {}
t0 = time.time()
for i, qid in enumerate(dev_ids):
    item = train_data[qid]
    ranked, scores = rrf_retrieve(item["question"], bm25, DENSE_CHANNELS, all_chunks, return_scores=True)
    rrf_ranked_cache[qid] = ranked
    rrf_scores_cache[qid] = scores
    if (i + 1) % 50 == 0 or (i + 1) == len(dev_ids):
        print(f"    retrieval (RRF gaps) {i+1}/{len(dev_ids)} ... {time.time()-t0:.0f}s")

if recall_ids:
    hits = {k: 0 for k in ks}
    for qid in recall_ids:
        ranked_ids = [c["id"] for c in rrf_ranked_cache[qid]]
        pos_id = train_positive[qid]
        for k in ks:
            if pos_id in ranked_ids[:k]:
                hits[k] += 1
    nr = len(recall_ids)
    recall_at_k_by_label["BM25+dense (RRF gaps, không rerank)"] = {str(k): round(hits[k] / nr, 4) for k in ks}

ms, rs = [], []
for qid in dev_ids:
    ranked, scores = rrf_ranked_cache[qid], rrf_scores_cache[qid]
    k = adaptive_k_cutoff(scores) if ranked else 1
    pred = render_answer(ranked, k, train_data[qid]["question"]) if ranked else "Không tìm thấy thông tin pháp lý cho câu hỏi này."
    ref = train_data[qid]["answer"]
    ms.append(meteor_score([str(ref).split()], str(pred).split()))
    rs.append(rouge.score(str(ref), str(pred))["rougeL"].fmeasure)
m, r = sum(ms) / len(ms), sum(rs) / len(rs)
print(f"    adaptive-k (RRF gaps)  METEOR={m:.4f}  ROUGE-L={r:.4f}  (n={len(dev_ids)})")
best_use_rrf_gaps = False
if m > best_m:
    best_m, best_r, best_use_rerank, best_use_adaptive = m, r, False, True
    best_rr_source, best_use_rrf_gaps = None, True

print(f"  => chọn TOP_N_ANSWER={best_n}, dùng reranker={best_use_rerank} "
      f"(nguồn={best_rr_source}), dùng adaptive-k={best_use_adaptive}, "
      f"RRF-gaps={best_use_rrf_gaps} (METEOR={best_m:.4f})")
top_n_answer, use_reranker, use_adaptive = best_n, best_use_rerank, best_use_adaptive
use_rrf_gaps = best_use_rrf_gaps

# Chốt reranker_models/reranker_tokenizers DÙNG CHO BƯỚC 7 theo đúng cái thắng ở dev-eval —
# nếu fine-tuned thắng, GÁN LẠI reranker_models = bộ fine-tuned; nếu zero-shot thắng (hoặc
# không dùng reranker), GIỮ NGUYÊN reranker_models (đã là zero-shot mặc định). Giải phóng
# bộ không dùng tới để đỡ VRAM cho Bước 7.
if best_rr_source == "finetuned":
    print("  Reranker fine-tuned THẮNG dev-eval -> dùng cho Bước 7 (giải phóng bản zero-shot).")
    for dev, m in list(reranker_models.items()):
        del m
    reranker_models, reranker_tokenizers = reranker_models_ft, reranker_tokenizers_ft
elif HAS_RERANKER_FT:
    print("  Reranker zero-shot THẮNG (hoặc không dùng reranker) -> giải phóng bản fine-tuned, "
          "không dùng cho Bước 7.")
    for dev, m in list(reranker_models_ft.items()):
        del m
reranker_models_ft, reranker_tokenizers_ft = {}, {}
try:
    torch.cuda.empty_cache()
except Exception:
    pass

eval_info = {"meteor": round(best_m, 4), "rouge_l": (round(best_r, 4) if best_r is not None else None),
             "recall_at_k": recall_at_k_by_label, "n_dev": len(dev_ids), "reranker_source": best_rr_source,
             "rrf_gaps_used": best_use_rrf_gaps}
checkpoint("Xong dev-eval + Recall@k")


# Cell 13: Bước 7 — Sinh câu trả lời cho public-official.json (song song 2 GPU nếu dùng reranker)
print("=== Bước 7: Sinh câu trả lời cho public-official.json ===")
with open(PUBLIC_PATH, encoding="utf-8") as f:
    questions = json.load(f)
qids = list(questions.keys())

if use_reranker and len(RERANK_DEVICES) > 1:
    def _worker(chunk, dev, widx):
        out = {}
        for i, qid in enumerate(chunk):
            out[qid] = answer_question(questions[qid]["question"], bm25, DENSE_CHANNELS,
                                        all_chunks, top_n_answer,
                                        reranker_model=reranker_models[dev],
                                        reranker_tokenizer=reranker_tokenizers[dev],
                                        use_adaptive_k=use_adaptive)
            _progress_print("Bước 7", widx, i, len(chunk))
        return out
    answers = parallel_process(qids, _worker, RERANK_DEVICES, label="Bước 7")
else:
    rr_dev = RERANK_DEVICES[0] if (use_reranker and RERANK_DEVICES) else None
    rr_model = reranker_models[rr_dev] if rr_dev else None
    rr_tok = reranker_tokenizers[rr_dev] if rr_dev else None
    answers = {}
    for i, qid in enumerate(qids):
        answers[qid] = answer_question(questions[qid]["question"], bm25, DENSE_CHANNELS,
                                        all_chunks, top_n_answer, reranker_model=rr_model,
                                        reranker_tokenizer=rr_tok, use_adaptive_k=use_adaptive)
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(qids)}  ({elapsed()/60:.1f} phút)")

n_empty = sum(1 for a in answers.values() if not a.strip())
print(f"  Đã sinh {len(answers)} câu trả lời, {n_empty} câu rỗng")
checkpoint("Xong sinh câu trả lời")

# Cell 14: Bước 8 — Validate + đóng gói submission.zip (vào /kaggle/working) + ghi sổ thí nghiệm
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


print("=== Bước 8: Đóng gói submission.zip ===")
build_submission(answers, set(questions.keys()), Path(OUT_DIR) / "submission.zip")
checkpoint(f"XONG — tổng thời gian {elapsed()/60:.1f} phút (trần an toàn {TIME_BUDGET_SEC/3600:.0f} giờ)")

# BẢN SỬA — "sổ thí nghiệm": 1 dòng JSON/lần chạy, APPEND vào EXPERIMENT_LOG_PATH (nằm ở
# /kaggle/working nên được giữ lại khi "Save Version", khác cache/ ở /kaggle/temp bị xoá khi
# session kết thúc). Đủ để so sánh nhiều lần chạy sau này (khác seed, khác USE_WARMUP, khác
# USE_FINETUNE...) mà không phải lục lại log console — xem PHAN_TICH_KY_THUAT.md §5.8.
n_empty = sum(1 for a in answers.values() if not a.strip())
record = {
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "use_warmup": USE_WARMUP,
    "n_warmup_used": n_warmup_used,
    "hardware": (f"kaggle_t4x{N_GPU}" if IS_KAGGLE else f"local_{N_GPU}gpu"),
    "concl": CONCL, "use_task1_labels": USE_TASK1_LABELS, "n_task1_pairs_added": n_task1_added,
    "n_train_pairs_available": finetune_info["n_pairs_available"],
    "n_train_pairs_used": finetune_info["n_pairs_used"],
    "used_finetune": finetune_info["used_finetune"], "finetune_reason": finetune_info["reason"],
    "finetune_models": finetune_info["models"],  # {"bge-m3": {max_steps,...}, "e5-large": {...}}
    "reranker_finetuned": reranker_finetune_info["used"],
    "reranker_finetune_steps": reranker_finetune_info["steps"],
    "reranker_finetune_elapsed_min": round(reranker_finetune_info["elapsed_s"] / 60, 1),
    "reranker_finetune_n_pairs": len(rows_clean),  # CHỈ citation — Task 1 bị lọc, xem Cell 8
    "checkpoint_dir": CHECKPOINT_DIR if (finetune_info["used_finetune"] or reranker_finetune_info["used"]) else None,
    "top_n_answer": top_n_answer, "use_reranker": use_reranker, "use_adaptive_k": use_adaptive,
    "dev_meteor": eval_info["meteor"], "dev_rouge_l": eval_info["rouge_l"],
    "dev_n": eval_info["n_dev"], "dev_recall_at_k": eval_info["recall_at_k"],
    "n_empty_answers": n_empty, "elapsed_min": round(elapsed() / 60, 1),
}
try:
    with open(EXPERIMENT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Đã ghi thêm 1 dòng vào {EXPERIMENT_LOG_PATH} (sổ thí nghiệm — không ghi đè, "
          f"giữ lại khi Save Version).")
except OSError as e:
    print(f"  [CẢNH BÁO] Không ghi được sổ thí nghiệm ({e}) — không ảnh hưởng submission.zip.")
# SỬA (mất log thật: session Kaggle bị reset, /kaggle/working và /kaggle/temp mất theo, không
# đọc lại được experiment_log.jsonl): LUÔN in JSON đầy đủ ra console dù ghi file thành công hay
# không. Nếu bạn dùng "Save Version" > "Save & Run All (Commit)", output cell này được giữ lại
# trong tab Output/Logs của version đó ngay cả khi bạn reset editor sau này — an toàn hơn nhiều
# so với chỉ trông vào file trong /kaggle/working, vốn cũng có thể mất nếu không commit đúng cách.
print("  [SỔ THÍ NGHIỆM — copy dòng dưới đây nếu cần đối chiếu sau này]")
print("  " + json.dumps(record, ensure_ascii=False))