# 📊 BÁO CÁO KỸ THUẬT – LEGALQA (UIT DSC2026 TASK 2)

## ===== ĐIỂM SỐ =====
**METEOR: 0.3906**
*(Đạt được trên tập public-official với pipeline tối ưu cho RTX 2050 4GB VRAM)*

---

## 📋 TÓM TẮT LUỒNG XỬ LÝ

```
INPUT: Câu hỏi pháp luật tiếng Việt
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 1: CHUNKING & BM25 INDEXING                       │
│ - Chia văn bản theo Điều (không cắt độ dài)            │
│ - Trích xuất metadata (loai_vb, so_hieu)               │
│ - Xây dựng BM25 index (lexical retrieval)              │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 2: TRAINING DATA GENERATION                       │
│ - Trích xuất citation (Điều + so_hieu) từ train.json   │
│ - Ánh xạ → chunk ID (positive pairs)                   │
│ - Hard negative mining via BM25                        │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 3: DENSE RETRIEVER (FINE-TUNE)                    │
│ - Load PhoBERT-based SentenceTransformer               │
│ - Fine-tune với MultipleNegativesRankingLoss           │
│ - Time-boxed (tối đa 22 phút)                          │
│ - Batch size tự điều chỉnh theo VRAM                   │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 4: ENCODE CORPUS                                  │
│ - Encode toàn bộ chunk → embeddings vector              │
│ - Lưu cache để tái sử dụng                             │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 5: HYBRID RETRIEVAL (RRF)                         │
│ - BM25 (lexical) + Dense (semantic)                    │
│ - Reciprocal Rank Fusion → top‑100                     │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 6: ANSWER GENERATION                              │
│ - Template extractive (ghép nguyên văn top‑3/5 chunks)│
│ - Dùng loai_vb + so_hieu thật từ metadata             │
│ - Tối ưu cho METEOR (alpha=0.9 nặng recall)           │
└──────────────────────────────────────────────────────────┘
    │
    ▼
OUTPUT: submission.zip (1000 câu trả lời)
```

---

## 🔧 CÔNG NGHỆ SỬ DỤNG

| Phase | Công nghệ/Thư viện | Mục đích | Nguồn |
|-------|-------------------|----------|-------|
| **Chunking** | Regex + Python | Tách văn bản theo cấu trúc Điều (pháp lý) | Tự phát triển |
| **BM25** | NumPy + BM25 tự viết | Tìm kiếm lexical (khớp từ khoá) | Robertson & Zaragoza, 2009 |
| **Dense Retriever** | SentenceTransformers + PhoBERT | Embedding ngữ nghĩa | Reimers & Gurevych, 2019 |
| **Fine-tune** | MultipleNegativesRankingLoss | Huấn luyện dense retriever | Henderson et al., 2017 |
| **Hybrid Fusion** | RRF (Reciprocal Rank Fusion) | Kết hợp lexical + semantic | Cormack et al., 2009 |
| **Answer Gen** | Template Extractive | Sinh câu trả lời từ nguyên văn | Tự phát triển (theo SCORING_LegalQA.md) |
| **Time-boxing** | Tự động điều chỉnh số step | Đảm bảo < 1 giờ chạy | Tự phát triển |
| **Cache** | Pickle (.pkl) | Lưu embeddings, tokenized corpus | Tự phát triển |

---

## 🤖 MÔ HÌNH SỬ DỤNG

### 1. PhoBERT-based SentenceTransformer

| Thông tin | Chi tiết |
|-----------|----------|
| **Tên mô hình** | `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` |
| **Kiến trúc** | PhoBERT-base (135M tham số) + SimCSE fine-tune |
| **Loại** | Sentence Embedding Model |
| **Link HuggingFace** | [VoVanPhuc/sup-SimCSE-VietNamese-phobert-base](https://huggingface.co/VoVanPhuc/sup-SimCSE-VietNamese-phobert-base) |
| **Paper** | **SimCSE: Simple Contrastive Learning of Sentence Embeddings** — Gao et al., EMNLP 2021 |
| **Paper Link** | https://aclanthology.org/2021.emnlp-main.552/ |
| **Ứng dụng** | Dense Retriever (encode chunk và câu hỏi thành vector) |

---

### 2. BM25 (Lexical Retriever)

| Thông tin | Chi tiết |
|-----------|----------|
| **Phương pháp** | BM25 (Best Match 25) — tự viết bằng NumPy |
| **Loại** | Sparse Retrieval (tìm kiếm từ khoá) |
| **Paper** | **The Probabilistic Relevance Framework: BM25 and Beyond** — Robertson & Zaragoza, 2009 |
| **Paper Link** | https://link.springer.com/article/10.1007/s10791-009-9115-x |
| **Ứng dụng** | Lexical retrieval, hard negative mining, hybrid fusion |

---

### 3. RRF (Reciprocal Rank Fusion)

| Thông tin | Chi tiết |
|-----------|----------|
| **Phương pháp** | Reciprocal Rank Fusion (k=60) |
| **Loại** | Score fusion (kết hợp BM25 + Dense) |
| **Paper** | **Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods** — Cormack et al., SIGIR 2009 |
| **Paper Link** | https://dl.acm.org/doi/10.1145/1571941.1572114 |
| **Ứng dụng** | Kết hợp điểm từ BM25 và dense retriever → top‑100 |

---

### 4. MultipleNegativesRankingLoss

| Thông tin | Chi tiết |
|-----------|----------|
| **Phương pháp** | Multiple Negatives Ranking Loss (in-batch negatives) |
| **Loại** | Contrastive Learning Loss |
| **Paper** | **Efficient Natural Language Response Suggestion for Smart Reply** — Henderson et al., 2017 |
| **Paper Link** | https://arxiv.org/abs/1705.00652 |
| **Ứng dụng** | Fine‑tune dense retriever với positive + hard negative |

---

### 5. Tokenizer (tiếng Việt)

| Thông tin | Chi tiết |
|-----------|----------|
| **Phương pháp** | Regex tokenizer: `[^\W\d_]+\|\d+` |
| **Loại** | Tokenization cơ bản (không dùng PyVi) |
| **Ứng dụng** | BM25 indexing và query tokenization |

---

### 6. Word Segmentation (dùng trong chunking)

| Thông tin | Chi tiết |
|-----------|----------|
| **Phương pháp** | `pyvi.ViTokenizer` (đã tích hợp sẵn trong môi trường) |
| **Paper** | **pyvi: Python Vietnamese Toolkit** — Nguyễn et al. (không có paper chính thức) |
| **Link** | https://pypi.org/project/pyvi/ |
| **Ứng dụng** | Chunking và tokenization (nếu có) |

---

### 7. NLTK (METEOR/ROUGE-L Evaluation)

| Thông tin | Chi tiết |
|-----------|----------|
| **Phương pháp** | NLTK word_tokenize + PorterStemmer |
| **Loại** | Tokenization cho độ đo METEOR |
| **Paper** | **METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments** — Banerjee & Lavie, ACL 2005 |
| **Paper Link** | https://aclanthology.org/W05-0909/ |
| **Ứng dụng** | Dev‑eval chọn TOP_N_ANSWER |

---

### 8. ROUGE-L (Độ đo phụ)

| Thông tin | Chi tiết |
|-----------|----------|
| **Phương pháp** | RougeScorer (ROUGE-L) |
| **Loại** | Longest Common Subsequence |
| **Paper** | **ROUGE: A Package for Automatic Evaluation of Summaries** — Lin, 2004 |
| **Paper Link** | https://aclanthology.org/W04-1013/ |
| **Ứng dụng** | Đánh giá chất lượng câu trả lời (độ đo phụ) |

---

## 💡 Ý TƯỞNG XỬ LÝ CHÍNH

### A. Không dùng LLM sinh câu trả lời

| Vấn đề | Giải pháp | Lợi ích |
|--------|-----------|---------|
| LLM (1.5B+) + optimizer > 4GB VRAM | Dùng template extractive (ghép nguyên văn) | Tiết kiệm VRAM, tối ưu METEOR |
| Diễn giải lại làm giảm METEOR | Trích nguyên văn theo đúng thứ tự | Giảm penalty phân mảnh (mũ 3) |

---

### B. Fine-tune dense retriever nhưng time‑boxed

| Vấn đề | Giải pháp | Lợi ích |
|--------|-----------|---------|
| Fine-tune nhiều epoch > 1 giờ | Đo tốc độ step đầu → tính max_steps vừa ngân sách | Đảm bảo < 22 phút cho Bước 4 |
| 3565 positive pairs quá nhiều | Lấy mẫu ngẫu nhiên 1000 cặp | Giảm thời gian mining hard‑negative |

---

### C. Tự điều chỉnh batch size khi OOM

| Vấn đề | Giải pháp | Lợi ích |
|--------|-----------|---------|
| 4GB VRAM dễ OOM với batch=8 | Thử batch=8 → nếu OOM, giảm xuống 4 → 2 | Chạy ổn định trên RTX 2050 |

---

### D. Tắt các cuộc gọi mạng để tránh treo

| Vấn đề | Giải pháp | Lợi ích |
|--------|-----------|---------|
| HF Hub gọi mạng (telemetry) gây treo | `HF_HUB_DISABLE_TELEMETRY=1` `HF_HUB_OFFLINE=1` | Không chờ mạng → pipeline nhanh hơn |

---

### E. In progress bar rõ ràng

| Vấn đề | Giải pháp | Lợi ích |
|--------|-----------|---------|
| Không in log → tưởng treo | In tiến trình mỗi 500 câu trong `_build_training_rows` | Biết pipeline đang chạy, không phải treo |

---

### F. Sinh nhãn từ citation trong train.json

| Vấn đề | Giải pháp | Lợi ích |
|--------|-----------|---------|
| Không có nhãn mức chunk trong train | Trích `Điều X` + `so_hieu` → map sang chunk | Tạo positive pairs tự động (3565 cặp) |

---

## 📚 TÀI LIỆU THAM KHẢO

| Công nghệ | Paper/Tài liệu |
|-----------|----------------|
| **PhoBERT** | Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. *Findings of EMNLP 2020*. |
| **SimCSE** | Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. *EMNLP 2021*. |
| **SentenceTransformers** | Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*. |
| **BM25** | Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*. |
| **RRF** | Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR 2009*. |
| **MultipleNegativesRankingLoss** | Henderson, M. et al. (2017). Efficient Natural Language Response Suggestion for Smart Reply. *arXiv:1705.00652*. |
| **METEOR** | Banerjee, S., & Lavie, A. (2005). METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. *ACL 2005*. |
| **ROUGE** | Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *ACL 2004*. |

---

## 📦 THƯ VIỆN CÀI ĐẶT

```bash
pip install numpy sentence-transformers datasets accelerate nltk rouge_score
```

**Lưu ý:** Cần cài torch bản CUDA (không dùng bản CPU-only):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```
