# Run EDA Analysis

Một file duy nhất, tất cả analyses.

## Chạy

```bash
python eda_all_in_one.py --data data/LegalQA_Public_Test --contexts selected-contexts
```

## Output

Folder `eda_output/` sẽ có 6 files:

1. **eda_qa_analysis.json** — 7000 QA pairs, chi tiết từng item
2. **eda_doc_analysis.json** — documents, chi tiết từng item
3. **eda_patterns_summary.txt** — text summary
4. **eda_chunking_comparison.json** — so sánh 4 strategies trên toàn bộ docs
5. **eda_chunking_stats.txt** — text summary chunking
6. **eda_samples.json** — 50+50 QA samples + 30+30 doc samples

## Thời gian

~20-30 phút (phụ thuộc vào SSD/CPU)

## Explore Results

```python
import json

# QA analysis
with open("eda_output/eda_qa_analysis.json") as f:
    qa = json.load(f)

# % có Điều
pct = 100 * sum(1 for q in qa if q["has_dieu"]) / len(qa)
print(f"{pct:.1f}% QA có Điều")

# Chunking
with open("eda_output/eda_chunking_comparison.json") as f:
    chunks = json.load(f)

doc = chunks[0]
print(f"Fixed: {doc['strategies']['Fixed 450W']['num_chunks']} chunks")
print(f"Hybrid: {doc['strategies']['Hybrid']['num_chunks']} chunks")

# Samples
with open("eda_output/eda_samples.json") as f:
    samples = json.load(f)
print(samples["metadata"])
```
