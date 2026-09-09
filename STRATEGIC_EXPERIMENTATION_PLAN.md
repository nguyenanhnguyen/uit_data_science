# Strategic Experimentation Plan — Tối ưu ROI mỗi lần train

**Context:** Tài nguyên giới hạn (GPU/time), hạn deadline 18/09/2026, muốn rút kinh nghiệm tối đa mỗi lần chạy.

**Nguyên tắc:**
- Mỗi lần train = 1 batch test với 3–5 hypothesis kết hợp (không tuỳ ý thay đổi)
- Control/treatment rõ ràng → rút được learnings có thể transfer
- Loại bỏ hypothesis đã closed / quá rủi cao
- Ưu tiên hypothesis **có dữ liệu hỗ trợ từ EDA hoặc result.md**

---

## 1. Audit: Những gì đã thử + đã đóng (từ result.md)

### Đã đóng (không làm lại):
✅ **Fine-tune encoder/reranker thêm lần**
- Từ #4 → #5: thêm dense encoder (bge-m3 + e5-large) → +0,0576 METEOR
- Từ #5 → #6: fine-tune reranker → +0,0313 METEOR (nhẹ)
- Hướng: đã cố định (e5-large + bge-m3 + zero-shot Vietnamese_Reranker), không thêm được

✅ **Ghép nhiều Điều (top_n > 1)**
- Oracle: 1 Điều = 0,605 METEOR; 2 Điều = 0,519 (MẤT 0,086)
- Kết luận: top_n=1 là tối ưu, không test top_n>1

✅ **Dùng LLM generator**
- Task 2 là extraction-based (METEOR α=0.9 nặng recall), sinh mới câu sẽ thua trích xuất

✅ **Cắt theo Điều ở tầng retrieval**
- Task 1 đo: làm hại -1,19 điểm recall
- EDA: 11,5% document tạo chunk >5.000 từ, 2,3% >20.000 từ
- Kết luận: Fixed 450W ở tầng 1 là an toàn, không test Articles ở tầng 1

✅ **Thêm epoch/LR/negatives ở fine-tune**
- Đã đo, không tăng thêm được
- Hiện config cố định

### Còn mở (đáng test):
- 🟡 **Mở rộng tầng 2: regex Mục/Phần/tiết** (15,5% document không Điều) — **Scenario A**
- 🟡 **Hybrid chunking ở tầng 1** (tôn trọng câu tốt hơn) — **Scenario B**
- 🟡 **Query expansion** (tăng từ khoá) — **Scenario C**
- 🟡 **Cắt câu trước trích** (post-processing) — **Scenario D**

---

## 2. Hypothesis Priority Matrix

| Hypothesis | Công sức | Rủi ro | Tác động ước tính | Bằng chứng | **Priority** |
|---|---|---|---|---|---|
| **A1: Mở rộng tầng 2 (MUC_RE)** | 2–3h | 🟢 Rất thấp | +0,3–0,8 | EDA phát hiện 15,5% docs |  **⭐⭐⭐ NGAY** |
| **A2: Chặn trên Điều >2000 từ** | 0,5h | 🟢 Rất thấp | +0,1–0,2 | EDA: outlier catastrophic | **⭐⭐⭐ NGAY** |
| **B1: Thử Hybrid ở tầng 1** | 3–4h + encode | 🟡 Trung | +0,2–0,3 | result.md §3 + EDA | **⭐⭐ (sau A)** |
| **C1: Query expansion** | 2–3h | 🟡 Trung | +0,2–0,5 | result.md §10: +0,1–0,2 | **⭐⭐ (nếu còn)** |
| **D1: Cắt câu pre-retrieval** | 1–2h | 🟡 Trung | +0,05–0,15 | chưa thử | **⭐ (nếu còn rất nhiều)** |
| **Reranker ensemble** | 4–6h | 🔴 Cao | +0,3–0,8 | chưa thử, rủi cao | **❌ SKIP** |
| **Thêm LLM synthetic data** | 6–8h | 🔴 Cao | ? | LLM sinh có thể ko match gold | **❌ SKIP** |

---

## 3. Design of Experiments (DOE) — Batch Testing

### Batch 1 (NGAY, 4–5 giờ): A1 + A2 + kiểm chứng

**Hypothesis ghép:**
```
Batch 1: Tầng 2 mở rộng + chặn trên + kiểm chứng
├─ Hypothesis A1: MUC_RE + PHU_LUC_RE + fallback tiết (TCVN)
├─ Hypothesis A2: Chặn trên nếu 1 Điều >2000 từ → fallback Fixed 450W
└─ Control: Fixed 450W (baseline hiện tại)
```

**Kiểu test:** A/B — Batch 1 vs Fixed 450W hiện tại
- Chỉ đổi tầng 2, giữ nguyên: tầng 1 (Fixed 450W), tầng 3 (echo2), encoder/reranker (cố định)
- Đo: `eval/score_dev_v2.py` trên 501 dev

**Expected output:**
- ✅ Nếu +0,1–0,8 → nộp public (dùng 1 lượt), quyết định nộp private hay không
- ❌ Nếu ≤0 → skip B1/C1, xem D1 hoặc dừng

**Lý do ghép A1+A2:** không phụ thuộc nhau, cùng quy tắc mở rộng tầng 2, có thể test cùng lúc

---

### Batch 2 (nếu Batch 1 tốt + còn GPU, 4 giờ): B1 + kiểm chứng

**Hypothesis:**
```
Batch 2: Hybrid ở tầng 1
├─ Hypothesis B1: Fixed 450W → Hybrid (tôn trọng câu)
└─ Control: Fixed 450W (từ Batch 1 kết quả)
```

**Kiểu test:** A/B — Hybrid vs Fixed 450W
- Thay đổi: tầng 1 retrieval chunking
- Phải: encode lại corpus (dense embedding) ~2–3h GPU
- Đo: dev-eval sau encode xong

**Expected output:**
- ✅ Nếu +0,1–0,3 → có thể nộp (nhưng cần kiểm chứng, rủi ko cao lắm)
- ⚠️ Nếu 0 ~ ±0,05 → borderline, xem còn bao nhiêu lượt submit + thời gian
- ❌ Nếu <-0,05 → roll back

**Lý do không ghép với A1:** 
- Batch 1 là code change nhẹ (regex), Batch 2 là encode lại corpus (heavy)
- Muốn isolate A1 tác động trước khi test B1
- Nếu ghép sẽ không biết tác động nào từ A1 vs B1

---

### Batch 3 (nếu A+B still khó, 3–4 giờ): C1 + kiểm chứng

**Hypothesis:**
```
Batch 3: Query expansion
├─ Hypothesis C1: Thêm từ khoá từ dense → mở rộng query
└─ Control: Query gốc (không expand)
```

**Kiểu test:** A/B — expand vs không
- Không cần encode lại corpus (chỉ expand ở retrieval time)
- Đo: dev-eval

**Expected output:**
- ✅ Nếu +0,05–0,15 → có thể nộp (nhẹ nhưng ổn)
- ❌ Nếu ≤0 → skip

**Lý do để sau A/B:** Đã có tiền lệ result.md §10 ghi +0,1–0,2, chắc chắn hơn nhưng lợi nhỏ hơn A/B

---

### Batch 4 (nếu vẫn còn rất nhiều, 2–3 giờ): D1

**Hypothesis:**
```
Batch 4: Pre-retrieval sentence cut
├─ Hypothesis D1: Cắt câu trước retrieval → chunk lõi (sentence-based)
└─ Control: Fixed 450W
```

**Kiểu test:** A/B — pre-cut vs không
- Cắt câu (`.!?`) → lọc chunk không nằm giữa câu → re-encode dense nếu cần
- Hoặc: chỉ filter candidate ở retrieval time (không encode lại)

**Expected output:**
- ✅ Nếu +0,03–0,1 → có thể nộp
- ❌ Nếu ≤0 → stop

---

## 4. Kiểm chứng & Transfer Learning

### Mỗi batch sau khi chạy:

1. **Không chỉ nhìn METEOR:** xem thêm:
   - Recall@1, @3, @5 (câu có document đúng trong top-K)
   - Độ dài answer trung bình (có đổi so với baseline không?)
   - % câu mà "không Điều" (15,5%) có được cải thiện hay không?

2. **Save output detail:**
   ```python
   # Lưu kết quả per-QA để phân tích sau
   {
     "qid": "...",
     "question": "...",
     "gold_answer": "...",
     "predicted_answer": "...",
     "meteor_this": 0.XX,
     "meteor_baseline": 0.YY,
     "retrieved_dieu": "...",  # Điều/Mục đã trích
     "retrieval_rank": 5,      # rank của doc đúng
   }
   ```

3. **Rút kinh nghiệm để transfer:**
   - Nếu Batch 1 (A1) tốt: các heuristic nào khớp được? Có general-purpose không?
   - Nếu Batch 2 (B1) tốt/xấu: có bài học gì về retrieval chunking cho Task 1?
   - Nếu C1 tốt: có combine được với A1/B1 không? (hybrid A+C)

---

## 5. Cách chạy mỗi Batch (workflow chi tiết)

### Batch 1 (A1 + A2):

```bash
# Tạo branch riêng
git checkout -b feature/tage2-muc-phu-luc

# Code A1: thêm MUC_RE, PHU_LUC_RE
# Code A2: chặn trên Điều > 2000

# Test trên vài sample từ eda_samples.json (quick validation)
python test_stage2_regex.py --samples eda_server_output/eda_samples.json

# Chạy dev-eval
python legalqa_local.py --data data/LegalQA_Public_Test \
  --contexts selected-contexts \
  --output-dir /tmp/batch1_output

# So sánh với baseline
python compare_metrics.py --baseline <path_to_baseline_result> \
  --new /tmp/batch1_output/metrics.json

# Nếu tốt:
git commit -m "feat: Stage 2 expansion MUC_RE + PHU_LUC_RE + bound check"
# Nếu xấu:
git reset --hard origin/main
```

### Batch 2 (B1):

```bash
# Tạo branch riêng
git checkout -b feature/hybrid-chunking-stage1

# Code B1: thay split_fixed → split_hybrid

# Encode lại corpus (GPU task ~2–3 giờ)
python legalqa_local.py --data ... --encode-only \
  --chunking hybrid \
  --output-dir /tmp/hybrid_encoded

# Chạy dev-eval (dùng cached embedding)
python legalqa_local.py --data ... \
  --use-cache /tmp/hybrid_encoded \
  --output-dir /tmp/batch2_output

# Compare
python compare_metrics.py --baseline batch1_result \
  --new /tmp/batch2_output

# Decision: nộp, skip, hoặc combine với batch 1?
```

---

## 6. Lộ trình triển khai (theo timeline)

| Ngày | Công việc | GPU time | Expected outcome |
|---|---|---|---|
| **9/9 (hôm nay)** | Batch 1 code + test quick | 1–2h GPU | Biết A1 tác động |
| **10/9** | Batch 1 dev-eval chạy xong | 30 min | Quyết định nộp A1 hay không |
| **11–12/9** | Batch 2 (B1) nếu A1 tốt | 3–4h GPU | Biết B1 tác động |
| **13/9** | Batch 3 (C1) nếu vẫn khô | 2h GPU | Cuối cùng thử |
| **14–18/9** | Nộp private test + optimize cuối | — | Final submission |

**Điểm dừng an toàn:** nếu sau Batch 1 không tốt (≤0), có thể roll back, xem Batch 3/4 nhẹ hơn, hoặc chuẩn bị nộp config hiện tại (v4: 0.5585).

---

## 7. Giả sử từng Batch không tốt — Plan B

### Nếu Batch 1 (A1) ≤0:
- Có thể hiện tại retrieval/reranker đã "bão hoà" — tầng 2 extraction không phải병kiếm
- **Plan B:** chuyển sang Batch 3 (C1 query expansion) hoặc tạm dừng
- **Lý do:** A1 có dữ liệu rõ ràng từ EDA (15,5% document không Điều), nếu cảm thấy không hiệu — có khả năng tầng extraction đã optimal

### Nếu Batch 2 (B1) ≤0:
- Hybrid không tốt hơn Fixed 450W (hoặc còn xấu hơn)
- **Plan B:** giữ Fixed 450W, chuyển C1 hoặc D1
- **Lý do:** retrieval stage khó adjust mà không ảnh hưởng xấu (lịch sử Task 1)

### Nếu cả A+B+C đều ≤0:
- Có thể model hiện tại đã near-optimal cho tập dữ liệu này
- **Plan B:** nộp config v4 hiện tại (0.5585), chuyển internal focus sang Task 1 hoặc clean-up code

---

## 8. Checkpoint lưu — ghi nhớ gì để transfer

Mỗi batch sau khi xong, ghi lại **1 page kinh nghiệm:**

```markdown
# Batch X Learnings

## Kết quả
- METEOR: ... (so với baseline)
- Hypothesis: tác động là ... (as expected / surprised)

## Chi tiết
- Regex MUC_RE khớp được bao nhiêu % query không-Điều?
- Fallback rate: bao nhiêu % query phải fallback Fixed 450W?
- Per-group analysis: nhóm "không Điều" có tăng bao nhiêu?

## Transfer
- Technique này có apply được cho Task 1 không?
- Có combine được với kỹ thuật khác không?
- Có generalize để future tasks không?

## Next priority (nếu tiếp tục)
- Vị trí rào cản tiếp theo là gì?
```

---

## 9. Tóm lại: Cách chạy tối ưu ROI

✅ **Làm ngay (Batch 1):** A1 (MUC_RE) + A2 (chặn 2000W)
- Dữ liệu: EDA phát hiện 15,5% document
- Rủi ro: rất thấp
- Công sức: 2–3h
- Tác động: +0,3–0,8 METEOR

✅ **Nếu Batch 1 tốt (Batch 2):** B1 (Hybrid)
- Dữ liệu: result.md + EDA
- Rủi ro: trung bình (cần encode lại)
- Công sức: 3–4h
- Tác động: +0,2–0,3 METEOR

✅ **Nếu vẫn khô (Batch 3):** C1 (Query expansion)
- Dữ liệu: result.md §10 ghi +0,1–0,2
- Rủi ro: trung bình
- Công sức: 2–3h
- Tác động: +0,2–0,5 METEOR

❌ **Không làm:** Reranker ensemble, LLM synthetic, fine-tune thêm
- Đã closed hoặc quá rủi cao

**Mục tiêu:** mỗi batch học được một điều, không thử bừa bãi.
