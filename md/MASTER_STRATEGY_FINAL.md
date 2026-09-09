# Master Strategy — Tổng hợp Path từ 0.4469 → 0.5585 METEOR

**Ngày:** 2026-09-09 · **Dataset:** 7.000 QA + 8.474 documents · **Công thức chấm:** METEOR α=0.9 (nặng recall)

---

## 1. Tiến hóa điểm số — Học được gì từng bước

| Ver | METEOR | ROUGE-L | Thay đổi chính | Tác động |
|---|---:|---:|---|---|
| **Baseline (0.4469)** | 0.4469 | 0.3801 | SimCSE + bkai encoder + fine-tune reranker | — |
| **v2–v3 (0.5215)** | 0.5215 | 0.4829 | +2 dense encoder khác họ (bge-m3 + e5-large) + RRF + zero-shot reranker | **+0.0746 METEOR** |
| **v6 (0.5528)** | 0.5528 | 0.4835 | +fine-tune reranker + echo2 template | +0.0313 (nhẹ) |
| **v7 (0.5557)** | 0.5557 | 0.5465 | Đổi encoder (Vietnamese_Embedding_v2 + harrier) + LSE aggregation | +0.0029 METEOR, **+0.0630 ROUGE-L** 🔴 |
| **v4 (0.5585)** | 0.5585 | 0.5419 | **2-tier architecture** (tầng 1: 450W chunk retrieval, tầng 2: extract Điều) + family_negatives | +0.0057 METEOR, -0.0046 ROUGE-L |

---

## 2. Phân tích: Tại sao v4 (0.5585) là tối ưu hiện tại

### 2.1 Kiến trúc 2-tầng của v4 (BREAKTHROUGH)

```
Tầng 1 (RETRIEVAL):     Question → [BM25 + bge-m3 + e5-large (RRF)]
                        × 450-word fixed chunks
                        ↓ Rerank → Top-5 DOCUMENTS
                        
Tầng 2 (EXTRACTION):    Mỗi document → [Split theo Điều]
                        × Rerank đơn vị Điều → Top-1 ĐIỀU
                        
Tầng 3 (COMPOSE):       Điều + Template (Căn cứ + câu dẫn + echo2)
```

**Tại sao tầng 1 dùng 450-word fixed chunking (không Điều)?**
- EDA phát hiện: 11,5% document tạo chunk >5.000 từ khi cắt Điều (outlier catastrophic)
- Task 1 (LegalIR) đo được: cắt theo Điều ở tầng retrieval → -1,19 điểm recall
- Lý do: embedding bị pha loãng khi 1 "Điều" = 50.000 từ (lỗi OCR/format)
- **Kết luận:** Tầng retrieval cần **chặn cứng max=450 từ** để tránh outlier

**Tại sao tầng 2 cắt theo Điều?**
- Oracle đo: 1 Điều trích → METEOR 0.605
- 72,2% câu chỉ cần 1–2 Điều → ghép thêm là giảm (2 Điều = 0.519)
- **Kết luận:** Tầng extraction cần độ phân giải cao (Điều) để trích đúng

**Tại sao v4 tốt hơn v7?**
- v7: ROUGE-L cao (0.5465) do **template khớp tốt trên train.json**
- v4: METEOR cao (0.5585) vì **architecture (2-tier) + family_negatives làm cải thiện retrieval + fine-tune reranker**
- v7 thử encoder mới (Vietnamese_Embedding_v2 + harrier) nhưng gặp fallback (encoder bị lỗi)
- **Verdict:** v4 architecture là tối ưu, không nên đổi encoder nếu chưa kiểm chứng

### 2.2 Kỹ thuật key trong v4

| Kỹ thuật | Từ đâu | Tác động |
|---|---|---|
| **2-tier architecture** | Logic analysis § 3 (legalqa/result.md) | +0.1–0.2 (estimate từ oracle) |
| **Family negatives** (rerank train) | Task 1 chuyển giao | +0.05–0.1 (hard negative mining) |
| **LSE aggregation** (doc-level gộp) | v7 thử, nhưng v4 KHÔNG dùng tầng 1 (chỉ dùng max ở tầng 1, LSE ở tầng 2) | ± 0 ở tầng 1 (unsafe với chunk lớn) |
| **Echo2 template** (lặp câu 2 lần) | v6 → v4 inherit | +0.0131 ± 0.0010 (nhẹ nhưng ổn) |
| **Fine-tune reranker** | v6 đầu tiên | +0.0313 METEOR (nhẹ) |
| **RRF 3-kênh** (BM25 + 2 dense) | v2–v3 | +0.0746 METEOR (lớn nhất) |

---

## 3. Những gì đã đóng — KHÔNG LÀM LẠI

✅ **Fine-tune encoder thêm lần**
- Đã cố định bge-m3 + e5-large, không tăng được
- v7 thử encoder mới (Vietnamese_Embedding_v2 + harrier) → fallback do lỗi → ROUGE-L cao nhưng METEOR nhẹ

✅ **Ghép nhiều Điều (top_n > 1)**
- Oracle: 1 Điều = 0.605, 2 Điều = 0.519 (MẤT 0.086)
- Hiện tại top_n=1 là tối ưu

✅ **Dùng LLM generator**
- Task 2 là extraction-based (METEOR α=0.9 nặng recall), LLM sinh mới sẽ thua

✅ **Thêm epoch/LR ở fine-tune**
- Đã đo, không tăng được nữa

✅ **Cắt theo Điều ở tầng 1**
- Outlier catastrophic (11,5% doc), không dùng ở retrieval

---

## 4. Chiến lược tiếp theo — Batch 1/2/3

### **Batch 1 (NGAY, 4–5h): Mở rộng tầng 2 — A1+A2**

**Phát hiện từ EDA:** 15,5% document không Điều nhưng vẫn có cấu trúc (Mục/Phần/tiết/Phụ lục)

**Code A1: MUC_RE + PHU_LUC_RE + fallback tiết**
```python
# Trong stage 2 extract, thêm regex tầng bậc sau Điều:
MUC_RE = r"(?m)^\s*Mục\s+(\d+|[IVXLCDM]+)\s*[.．:]"
PHU_LUC_RE = r"(?m)^\s*Phụ\s+lục\s+([IVXLCDM\d]+)"
TIET_RE = r"(?m)^\s*(\d+\.\d+(\.\d+)?)\s"  # TCVN dạng 2.2.1.2

# Logic: thử DIEU_RE → MUC_RE → PHU_LUC_RE → TIET_RE → fallback Fixed 450W
```

**Code A2: Chặn trên Điều >2000 từ**
```python
# Nếu 1 Điều bị tách ra >2000 từ (lỗi OCR) → fallback Fixed 450W cho riêng Điều đó
if max_dieu_len > 2000:
    dieu_chunks = split_fixed(dieu_text, 450)
```

**Tác động ước tính:** +0,3–0,8 METEOR (nếu regex khớp 70%+ của 15,5% doc)

**Rủi ro:** Rất thấp (chỉ tăng regex, không chạm 84,5% nhóm Điều)

---

### **Batch 2 (nếu A1 tốt, 4h): Thử Hybrid ở tầng 1**

**Giả sử A1 ✅ +0,3–0,5 METEOR → thử B1:**
- Thay Fixed 450W → Hybrid (450W nhưng tránh cắt giữa câu)
- Chunk avg từ 422,1 → 374,9 từ

**Tác động ước tính:** +0,2–0,3 METEOR (uncertain, cần đo A/B)

**Rủi ro:** Trung bình (cần encode lại corpus ~2–3h GPU)

**Điều kiện:** Chỉ làm nếu Batch 1 ✅ + còn GPU + còn lượt submit

---

### **Batch 3 (fallback, 2–3h): Query Expansion**

**Nếu A+B không đủ → thử C1:**
- Expand query từ dense encoder → thêm từ khoá
- result.md §10 ghi +0,1–0,2 METEOR

**Tác động ước tính:** +0,2–0,5 METEOR

---

## 5. Workflow triển khai cuối cùng

```bash
# Branch riêng cho Batch 1
git checkout -b feature/stage2-expansion

# A1: Code MUC_RE + PHU_LUC_RE + TIET_RE
# A2: Code chặn trên 2000W
# Test quick trên 5 sample từ eda_samples.json

python legalqa_local.py --data data/LegalQA_Public_Test \
  --contexts selected-contexts \
  --output-dir /tmp/batch1_output

python compare_metrics.py --baseline current_v4_result \
  --new /tmp/batch1_output/metrics.json

# Nếu METEOR +0,1–0,8 → nộp public test (dùng 1 lượt)
# Nếu ≤0 → roll back, xem Batch 3
```

---

## 6. Chunking mới trong Batch 1

### **Tầng 1 (Retrieval):** GIỮ NGUYÊN Fixed 450W
- Lý do: an toàn, đã tối ưu, chặn cứng max=450
- Alternative: Hybrid (nếu Batch 2) — tôn trọng câu tốt hơn

### **Tầng 2 (Extraction):** MỞ RỘNG regex
```
Hiện tại:  DIEU_RE → rỗng → fallback Fixed 450W
Batch 1:   DIEU_RE → MUC_RE → PHU_LUC_RE → TIET_RE → fallback Fixed 450W
```

**Tầu 3 (Compose):** GIỮ NGUYÊN echo2 template
- Lý do: đã tối ưu, +0,0131 điểm nhẹ nhưng ổn

---

## 7. Các kỹ thuật khác sẽ làm (nếu còn thời gian)

| Hướng | Công sức | Tác động | Priority |
|---|---|---|---|
| **Batch 2: Hybrid ở tầng 1** | 4h + encode | +0,2–0,3 | ⭐⭐ (sau A) |
| **Batch 3: Query expansion** | 2–3h | +0,2–0,5 | ⭐ (fallback) |
| ❌ Pre-retrieval sentence cut | 2–3h | +0,05–0,15 | (quá công sức, deadline sát) |
| ❌ Reranker ensemble | 4–6h | +0,3–0,8 | (quá rủi cao) |
| ❌ LLM synthetic data | 6–8h | ? | (vi phạm rule + extraction-based task) |

---

## 8. Tóm lại — Hướng làm tiếp (không lặp lại)

### **Từ main_version (v4 = 0.5585 là SOTA hiện tại):**
- ✅ 2-tier architecture (tầng 1: 450W retrieval, tầng 2: extract Điều) — **TỐI ƯU, GIỮ NGUYÊN**
- ✅ RRF 3-kênh (BM25 + bge-m3 + e5-large) — **TỐI ƯU, GIỮ NGUYÊN**
- ✅ Fine-tune reranker + echo2 — **TỐI ƯU, GIỮ NGUYÊN**
- ❌ Không dùng encoder mới (v7 thử Vietnamese_Embedding_v2 → fallback, không tốt)

### **Từ EDA (15,5% document không Điều, 11,5% chunk outlier):**
- ✅ A1: Mở rộng tầng 2 regex (MUC_RE + PHU_LUC_RE) — **PHẢI LÀM, có dữ liệu**
- ✅ A2: Chặn trên Điều >2000 từ — **PHẢI LÀM, xử lý outlier**
- ✅ B1: Thử Hybrid ở tầng 1 — **CÓ THỂ LÀMS (nếu A tốt + còn GPU)**

### **Những gì KHÔNG làm:**
- ❌ Fine-tune encoder thêm (đã cố định)
- ❌ Ghép nhiều Điều (oracle chứng minh MẤT điểm)
- ❌ LLM generator (extraction-based task)
- ❌ Cắt Điều ở tầng 1 (outlier catastrophic)

---

## 9. Checklist triển khai (Next 48h)

- [ ] Code A1 (MUC_RE + PHU_LUC_RE + TIET_RE)
- [ ] Code A2 (chặn 2000W)
- [ ] Test quick trên 5 sample
- [ ] Chạy score_dev_v2.py
- [ ] So sánh METEOR: baseline vs +A1+A2
- [ ] Nếu +0,1–0,8 → nộp public (1 lượt)
- [ ] Nếu ≤0 → xem Batch 3 (Query expansion)

**Timeline:**
- 09/09 hôm nay: code A1+A2
- 10/09: dev-eval A1+A2
- 11/09: quyết định nộp / chuyển Batch 2/3
- 14–18/09: nộp private test

**Deadline:** 18/09 23:59 GMT+7 (còn 9 ngày)
