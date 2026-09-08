# EDA Scripts — Comprehensive Data Analysis for LegalQA Chunking

Bộ 4 script để phân tích **toàn bộ data LegalQA**, không sample, output JSON để khám phá.

## Scripts

### 1. `eda_comprehensive.py` — Phân tích QA pairs + Documents

Phân tích:
- QA pairs: % có Điều, độ dài, loại văn bản, cấu trúc
- Documents: tỷ lệ Điều, độ dài, cấu trúc

**Output:**
- `eda_qa_analysis.json` (7000 items): chi tiết từng QA pair
- `eda_doc_analysis.json`: chi tiết từng document
- `eda_patterns_summary.txt`: báo cáo text

**Chạy:**
```bash
python eda_comprehensive.py \
  --data data/LegalQA_Public_Test \
  --contexts selected-contexts
```

**Thời gian:** ~5-10 phút (phụ thuộc vào SSD)

---

### 2. `eda_chunking_eval.py` — So sánh 4 chunking strategies

Chiến lược so sánh:
1. **Fixed 450W** — hiện tại, cắt cứng 450 từ
2. **Sentences** — cắt theo câu (., !, ?)
3. **Articles** — cắt theo Điều (nếu không có → fallback Fixed)
4. **Hybrid** — 450W nhưng tránh cắt giữa câu

**Output:**
- `eda_chunking_comparison.json`: chi tiết từng document cho 4 strategies
- `eda_chunking_stats.txt`: thống kê so sánh

**Chạy:**
```bash
python eda_chunking_eval.py --contexts selected-contexts
```

**Thời gian:** ~10-15 phút

---

### 3. `eda_sample_analysis.py` — Sample data để xem pattern

Lấy samples:
- 50 QA với Điều + 50 QA không Điều
- 30 documents có Điều + 30 không Điều

**Output:**
- `eda_samples.json`: 50+50 QA samples + 30+30 doc samples

**Chạy:**
```bash
python eda_sample_analysis.py \
  --data data/LegalQA_Public_Test \
  --contexts selected-contexts
```

**Thời gian:** ~2-3 phút

---

## Quick Start

```bash
cd /workingspace_aiclub/WorkingSpace/Personal/vannk/uit_dsc_2026/legalqa/

# Chạy tất cả (theo thứ tự)
python eda_comprehensive.py --data data/LegalQA_Public_Test --contexts selected-contexts
python eda_chunking_eval.py --contexts selected-contexts
python eda_sample_analysis.py --data data/LegalQA_Public_Test --contexts selected-contexts
```

**Tổng thời gian:** ~20 phút

---

## Xem Kết Quả

### 1. Báo cáo Text
```bash
# Tóm tắt patterns
cat eda_patterns_summary.txt

# Tóm tắt chunking
cat eda_chunking_stats.txt
```

### 2. Phân tích JSON

```python
import json

# QA analysis
with open("eda_qa_analysis.json") as f:
    qa_data = json.load(f)

# Lấy câu không Điều
no_dieu = [q for q in qa_data if not q["has_dieu"]]
print(f"% không Điều: {100*len(no_dieu)/len(qa_data):.1f}%")

# Xem vài sample không Điều
for item in no_dieu[:3]:
    print(f"Q: {item['question'][:80]}...")
    print(f"  Type: {item['primary_type']}")
    print(f"  Answer: {item['answer_first_100_chars']}...")

# Chunking comparison
with open("eda_chunking_comparison.json") as f:
    chunk_data = json.load(f)

# So sánh Fixed vs Hybrid
doc = chunk_data[0]
fixed = doc["strategies"]["Fixed 450W"]
hybrid = doc["strategies"]["Hybrid"]

print(f"Fixed: {fixed['num_chunks']} chunks, avg {fixed['avg_chunk_len']:.0f}W")
print(f"Hybrid: {hybrid['num_chunks']} chunks, avg {hybrid['avg_chunk_len']:.0f}W")
```

---

## Dự kiến Kết Quả

Dựa trên lanalys trước:

| Metric | Giá trị |
|--------|--------|
| QA pairs có Điều | ~85-90% |
| QA pairs không Điều | ~10-15% |
| Documents có Điều | ~84% |
| Documents không Điều | ~16% |
| Answer length (median) | ~280-300 từ |
| Document length (median) | ~600-700 từ |

**Khuyến nghị chunking:**
1. **Tầng 1 (Retrieval):** Dùng **Hybrid** (450W + tránh cắt câu)
2. **Tầng 2 (Extraction):** Cắt theo **Điều** (với fallback cho 16% docs)
3. **Fallback:**
   - Nếu không Điều → cắt theo khoản (1), (2), ...
   - Nếu không khoản → fixed 300-400 từ

---

## Troubleshooting

**Script chạy lâu?**
- Bình thường: 20-30 phút cho toàn bộ 8500 documents
- Kiểm tra CPU/IO không bị bottleneck

**OOM errors?**
- Scripts không load toàn bộ data vào memory
- Nếu vẫn OOM → chia thành chunks (thêm `--batch-size` nếu cần)

**Missing files?**
```bash
# Kiểm tra structure
ls -la data/LegalQA_Public_Test/
ls -la selected-contexts/ | head -20
```

---

## Next Steps

Sau khi chạy scripts và có output JSON:

1. **Phân tích chi tiết QA patterns** → thêm heuristic cho edge case
2. **Kiểm tra chunking quality** → tính recall@1 của từng strategy trên dev
3. **Implement fallback** → xử lý 15-16% edge case
4. **Đo dev-eval** → compare điểm METEOR của từng strategy

---

## File Reference

| File | Output |
|------|--------|
| `eda_comprehensive.py` | `eda_qa_analysis.json`, `eda_doc_analysis.json`, `eda_patterns_summary.txt` |
| `eda_chunking_eval.py` | `eda_chunking_comparison.json`, `eda_chunking_stats.txt` |
| `eda_sample_analysis.py` | `eda_samples.json` |

Tất cả output là **self-contained JSON**, có thể mở trong browser hay phân tích với Python/jq.
