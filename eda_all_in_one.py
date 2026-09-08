#!/usr/bin/env python3
"""
EDA All-In-One — Phân tích TOÀN BỘ data LegalQA, 1 file duy nhất.

Gộp tất cả analyses:
1. QA pairs + Documents analysis
2. Chunking strategies comparison
3. Sample extraction

Output: folder eda_output/
  - eda_qa_analysis.json
  - eda_doc_analysis.json
  - eda_patterns_summary.txt
  - eda_chunking_comparison.json
  - eda_chunking_stats.txt
  - eda_samples.json

Usage:
  python eda_all_in_one.py --data data/LegalQA_Public_Test --contexts selected-contexts
"""
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import statistics
import sys
import random

# ============================================================================
# CONSTANTS & REGEX
# ============================================================================
CHUNK_WORDS = 450
DIEU_RE = re.compile(r"\bĐiều\s+(\d+[a-zA-ZđĐ]?)\b")
DIEU_HEADING_RE = re.compile(r"(?m)^\s*Điều\s+(\d+[a-zA-ZđĐ]?)\s*[.．:]")
KHOANG_RE = re.compile(r"\b\((\d+)\)\s")
THONG_TU_RE = re.compile(r"Thông tư\s+(\d+[a-zA-Z]?/[\d/A-Z\-]+)")
QUYET_DINH_RE = re.compile(r"Quyết định\s+(\d+[a-zA-Z]?/[\d/A-Z\-]+|[\d/A-Z\-]+)")
NGHI_DINH_RE = re.compile(r"Nghị định\s+(\d+[a-zA-Z]?/[\d/A-Z\-]+|[\d/A-Z\-]+)")
LUAT_RE = re.compile(r"Luật\s+([^\n.;,]+)")


def count_words(text: str) -> int:
    return len(text.split())


def count_sentences(text: str) -> int:
    return len([s for s in re.split(r'[.!?]', text) if s.strip()])


# ============================================================================
# CHUNKING STRATEGIES
# ============================================================================
def split_fixed(text: str, n: int = 450):
    w = text.split()
    return [" ".join(w[i:i+n]) for i in range(0, len(w), n)] if w else [text]


def split_by_sentences(text: str, min_w: int = 200, max_w: int = 500):
    sents = re.split(r'([.!?])', text)
    chunks = []
    current = []
    curr_words = 0

    for sent in sents:
        sent_words = count_words(sent)
        if curr_words + sent_words <= max_w:
            current.append(sent)
            curr_words += sent_words
        else:
            if current:
                chunks.append("".join(current).strip())
            current = [sent]
            curr_words = sent_words

    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c] or [text]


def split_by_articles(text: str):
    ms = list(DIEU_HEADING_RE.finditer(text))
    if not ms:
        return split_fixed(text, 450)
    chunks = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        chunk = text[m.start():end].strip()
        chunks.append(chunk)
    return chunks


def split_hybrid(text: str, n: int = 450):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:min(i + n, len(words))]
        chunk_text = " ".join(chunk_words)
        last_sent = max(
            chunk_text.rfind('.'),
            chunk_text.rfind('!'),
            chunk_text.rfind('?')
        )
        if last_sent > len(chunk_text) * 0.7 and last_sent > 0:
            chunk_text = chunk_text[:last_sent + 1].strip()
            moved = count_words(chunk_text)
        else:
            moved = len(chunk_words)
        if chunk_text:
            chunks.append(chunk_text)
        i += moved
    return [c for c in chunks if c] or [text]


def eval_chunking(name: str, chunks: list, passage: str):
    if not chunks:
        return None
    chunk_lens = [count_words(c) for c in chunks]
    return {
        "strategy": name,
        "num_chunks": len(chunks),
        "avg_chunk_len": statistics.mean(chunk_lens),
        "min_chunk_len": min(chunk_lens),
        "max_chunk_len": max(chunk_lens),
        "stdev_chunk_len": statistics.stdev(chunk_lens) if len(chunk_lens) > 1 else 0,
    }


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def analyze_qa_pair(qid: str, qa: dict):
    ans = qa.get("answer", "")
    ans_len = count_words(ans)
    dieus = DIEU_RE.findall(ans)
    khoans = KHOANG_RE.findall(ans)

    types = {}
    if THONG_TU_RE.search(ans):
        types["Thông tư"] = len(THONG_TU_RE.findall(ans))
    if QUYET_DINH_RE.search(ans):
        types["Quyết định"] = len(QUYET_DINH_RE.findall(ans))
    if NGHI_DINH_RE.search(ans):
        types["Nghị định"] = len(NGHI_DINH_RE.findall(ans))
    if LUAT_RE.search(ans):
        types["Luật"] = len(LUAT_RE.findall(ans))

    primary_type = max(types.items(), key=lambda x: x[1])[0] if types else "Khác"

    return {
        "qid": qid,
        "question": qa.get("question", ""),
        "answer_len_words": ans_len,
        "answer_len_sentences": count_sentences(ans),
        "has_dieu": bool(dieus),
        "dieu_count": len(dieus),
        "dieus": dieus[:10],
        "has_khoang": bool(khoans),
        "khoang_count": len(khoans),
        "answer_types": types,
        "primary_type": primary_type,
        "answer_first_100_chars": ans[:100],
    }


def analyze_document(doc_id: str, doc: dict):
    passage = doc.get("passage") or ""
    passage_len = count_words(passage)
    dieus_heading = DIEU_HEADING_RE.findall(passage)

    return {
        "doc_id": doc_id,
        "name": doc.get("name", ""),
        "passage_len_words": passage_len,
        "has_dieu_structure": bool(dieus_heading),
        "dieu_count": len(dieus_heading),
        "dieus": dieus_heading[:20],
        "passage_first_200_chars": passage[:200],
    }


def extract_short_excerpt(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [+{len(text) - max_len} chars]"


# ============================================================================
# MAIN
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/LegalQA_Public_Test"))
    parser.add_argument("--contexts", type=Path, default=Path("selected-contexts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Create output dir
    out_dir = Path("eda_output")
    out_dir.mkdir(exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    train_path = args.data / "train.json"
    if not train_path.exists():
        print(f"❌ Không tìm {train_path}")
        sys.exit(1)

    print(f"Loading {train_path}...", end=" ", flush=True)
    with open(train_path, encoding="utf-8") as f:
        train = json.load(f)
    print(f"✓ {len(train)} QA pairs\n")

    context_files = sorted(args.contexts.glob("context_*.json"))
    if not context_files:
        print(f"❌ Không tìm context_*.json trong {args.contexts}")
        sys.exit(1)

    print(f"✓ {len(context_files)} context files\n")

    # ========================================================================
    # ANALYSIS 1: QA PAIRS + DOCUMENTS
    # ========================================================================
    print("=" * 80)
    print("1. ANALYZING QA PAIRS + DOCUMENTS")
    print("=" * 80)

    qa_results = []
    qa_stats = {
        "with_dieu": 0,
        "without_dieu": 0,
        "answer_lengths": [],
        "dieu_counts": [],
        "type_dist": Counter(),
        "answer_sentences": [],
    }

    for i, (qid, qa) in enumerate(train.items()):
        if (i + 1) % 1000 == 0:
            print(f"  QA {i + 1} / {len(train)}", flush=True)

        result = analyze_qa_pair(qid, qa)
        qa_results.append(result)

        qa_stats["answer_lengths"].append(result["answer_len_words"])
        qa_stats["answer_sentences"].append(result["answer_len_sentences"])
        qa_stats["type_dist"][result["primary_type"]] += 1

        if result["has_dieu"]:
            qa_stats["with_dieu"] += 1
            qa_stats["dieu_counts"].append(result["dieu_count"])
        else:
            qa_stats["without_dieu"] += 1

    # Save QA analysis
    qa_file = out_dir / "eda_qa_analysis.json"
    with open(qa_file, "w", encoding="utf-8") as f:
        json.dump(qa_results, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {qa_file.name}\n")

    # Analyze documents
    doc_results = []
    doc_stats = {
        "with_dieu": 0,
        "without_dieu": 0,
        "passage_lengths": [],
        "total_docs": 0,
    }

    print("Analyzing documents...")
    for i, fp in enumerate(context_files):
        if (i + 1) % 1000 == 0:
            print(f"  Doc {i + 1} / {len(context_files)}", flush=True)

        try:
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        passage = doc.get("passage") or ""
        if not passage or count_words(passage) < 100:
            continue

        doc_id = doc.get("id")
        result = analyze_document(doc_id, doc)
        doc_results.append(result)
        doc_stats["total_docs"] += 1

        doc_stats["passage_lengths"].append(result["passage_len_words"])

        if result["has_dieu_structure"]:
            doc_stats["with_dieu"] += 1
        else:
            doc_stats["without_dieu"] += 1

    # Save doc analysis
    doc_file = out_dir / "eda_doc_analysis.json"
    with open(doc_file, "w", encoding="utf-8") as f:
        json.dump(doc_results, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {doc_file.name}\n")

    # Save patterns summary
    total_qa = qa_stats["with_dieu"] + qa_stats["without_dieu"]
    summary_file = out_dir / "eda_patterns_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("QA PAIR STATISTICS:\n")
        f.write(f"  Total: {total_qa}\n")
        f.write(f"  With Điều: {qa_stats['with_dieu']} ({100*qa_stats['with_dieu']/total_qa:.1f}%)\n")
        f.write(f"  Without Điều: {qa_stats['without_dieu']} ({100*qa_stats['without_dieu']/total_qa:.1f}%)\n")
        f.write(f"  Avg answer length: {statistics.mean(qa_stats['answer_lengths']):.1f} words\n")
        f.write(f"  Median answer length: {statistics.median(qa_stats['answer_lengths']):.1f} words\n")
        f.write(f"  Avg sentences: {statistics.mean(qa_stats['answer_sentences']):.2f}\n\n")

        f.write("DOCUMENT STATISTICS:\n")
        if doc_stats["total_docs"] > 0:
            pct_with = 100 * doc_stats["with_dieu"] / doc_stats["total_docs"]
            pct_without = 100 * doc_stats["without_dieu"] / doc_stats["total_docs"]
            f.write(f"  Total: {doc_stats['total_docs']}\n")
            f.write(f"  With Điều: {doc_stats['with_dieu']} ({pct_with:.1f}%)\n")
            f.write(f"  Without Điều: {doc_stats['without_dieu']} ({pct_without:.1f}%)\n")
            f.write(f"  Avg passage length: {statistics.mean(doc_stats['passage_lengths']):.1f} words\n")
            f.write(f"  Median passage length: {statistics.median(doc_stats['passage_lengths']):.1f} words\n")
    print(f"  ✓ {summary_file.name}\n")

    # ========================================================================
    # ANALYSIS 2: CHUNKING STRATEGIES
    # ========================================================================
    print("=" * 80)
    print("2. EVALUATING CHUNKING STRATEGIES")
    print("=" * 80)

    chunk_results = []
    strategy_stats = defaultdict(lambda: {
        "num_chunks": [],
        "avg_lens": [],
        "min_lens": [],
        "max_lens": [],
        "stdevs": [],
    })

    docs_with_dieu_struct = 0
    docs_without_dieu_struct = 0

    for i, fp in enumerate(context_files):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1} / {len(context_files)}", flush=True)

        try:
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        passage = doc.get("passage") or ""
        if not passage or count_words(passage) < 100:
            continue

        doc_id = doc.get("id")

        if DIEU_HEADING_RE.search(passage):
            docs_with_dieu_struct += 1
        else:
            docs_without_dieu_struct += 1

        strategies = {
            "Fixed 450W": split_fixed(passage, 450),
            "Sentences": split_by_sentences(passage),
            "Articles": split_by_articles(passage),
            "Hybrid": split_hybrid(passage, 450),
        }

        doc_result = {
            "doc_id": doc_id,
            "passage_len": count_words(passage),
            "has_dieu": bool(DIEU_HEADING_RE.search(passage)),
            "strategies": {},
        }

        for strat_name, chunks in strategies.items():
            eval_result = eval_chunking(strat_name, chunks, passage)
            if eval_result:
                doc_result["strategies"][strat_name] = eval_result
                stats = strategy_stats[strat_name]
                stats["num_chunks"].append(eval_result["num_chunks"])
                stats["avg_lens"].append(eval_result["avg_chunk_len"])
                stats["min_lens"].append(eval_result["min_chunk_len"])
                stats["max_lens"].append(eval_result["max_chunk_len"])
                stats["stdevs"].append(eval_result["stdev_chunk_len"])

        chunk_results.append(doc_result)

    # Save chunking comparison
    chunk_file = out_dir / "eda_chunking_comparison.json"
    with open(chunk_file, "w", encoding="utf-8") as f:
        json.dump(chunk_results, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {chunk_file.name}\n")

    # Save chunking stats
    chunk_stats_file = out_dir / "eda_chunking_stats.txt"
    total_docs_analyzed = len(chunk_results)
    with open(chunk_stats_file, "w", encoding="utf-8") as f:
        f.write("CHUNKING STRATEGIES COMPARISON:\n")
        f.write(f"Total documents analyzed: {total_docs_analyzed}\n\n")

        for strat in ["Fixed 450W", "Sentences", "Articles", "Hybrid"]:
            stats = strategy_stats[strat]
            if stats["num_chunks"]:
                f.write(f"{strat}:\n")
                f.write(f"  Avg chunks/doc: {statistics.mean(stats['num_chunks']):.1f}\n")
                f.write(f"  Avg chunk length: {statistics.mean(stats['avg_lens']):.1f} words\n")
                f.write(f"  Chunk length stdev: {statistics.mean(stats['stdevs']):.1f}\n\n")

        if total_docs_analyzed > 0:
            pct = 100 * docs_without_dieu_struct / total_docs_analyzed
            f.write(f"DOCUMENT STRUCTURE:\n")
            f.write(f"  With Điều: {docs_with_dieu_struct} ({100*docs_with_dieu_struct/total_docs_analyzed:.1f}%)\n")
            f.write(f"  Without Điều: {docs_without_dieu_struct} ({pct:.1f}%)\n")
            f.write(f"  ⚠️  {pct:.1f}% will fallback in Articles strategy\n")
    print(f"  ✓ {chunk_stats_file.name}\n")

    # ========================================================================
    # ANALYSIS 3: SAMPLES
    # ========================================================================
    print("=" * 80)
    print("3. EXTRACTING SAMPLES")
    print("=" * 80)

    qa_with_dieu = [(qid, qa) for qid, qa in train.items() if DIEU_RE.search(qa.get("answer", ""))]
    qa_without_dieu = [(qid, qa) for qid, qa in train.items() if not DIEU_RE.search(qa.get("answer", ""))]

    samples_with = random.sample(qa_with_dieu, min(50, len(qa_with_dieu)))
    samples_without = random.sample(qa_without_dieu, min(50, len(qa_without_dieu)))

    qa_samples = {
        "with_dieu": [
            {
                "qid": qid,
                "question": qa["question"],
                "answer_len_words": count_words(qa["answer"]),
                "answer_excerpt": extract_short_excerpt(qa["answer"], 400),
                "dieus": DIEU_RE.findall(qa["answer"])[:5],
            }
            for qid, qa in samples_with
        ],
        "without_dieu": [
            {
                "qid": qid,
                "question": qa["question"],
                "answer_len_words": count_words(qa["answer"]),
                "answer_excerpt": extract_short_excerpt(qa["answer"], 400),
            }
            for qid, qa in samples_without
        ],
    }

    # Load documents for sampling
    docs_with_dieu_list = []
    docs_without_dieu_list = []

    for fp in context_files:
        try:
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        passage = doc.get("passage") or ""
        if not passage or count_words(passage) < 100:
            continue

        if DIEU_HEADING_RE.search(passage):
            docs_with_dieu_list.append((fp, doc))
        else:
            docs_without_dieu_list.append((fp, doc))

    samples_docs_with = random.sample(docs_with_dieu_list, min(30, len(docs_with_dieu_list)))
    samples_docs_without = random.sample(docs_without_dieu_list, min(30, len(docs_without_dieu_list)))

    doc_samples = {
        "with_dieu": [
            {
                "doc_id": doc.get("id"),
                "name": doc.get("name"),
                "passage_len_words": count_words(doc.get("passage", "")),
                "passage_excerpt": extract_short_excerpt(doc.get("passage", ""), 400),
                "dieus": DIEU_HEADING_RE.findall(doc.get("passage", ""))[:20],
                "num_dieus": len(DIEU_HEADING_RE.findall(doc.get("passage", ""))),
            }
            for fp, doc in samples_docs_with
        ],
        "without_dieu": [
            {
                "doc_id": doc.get("id"),
                "name": doc.get("name"),
                "passage_len_words": count_words(doc.get("passage", "")),
                "passage_excerpt": extract_short_excerpt(doc.get("passage", ""), 400),
            }
            for fp, doc in samples_docs_without
        ],
    }

    samples = {
        "qa_samples": qa_samples,
        "doc_samples": doc_samples,
        "metadata": {
            "total_qa_pairs": len(train),
            "total_documents": len(context_files),
            "qa_with_dieu": len(qa_with_dieu),
            "qa_without_dieu": len(qa_without_dieu),
            "docs_with_dieu": len(docs_with_dieu_list),
            "docs_without_dieu": len(docs_without_dieu_list),
        },
    }

    samples_file = out_dir / "eda_samples.json"
    with open(samples_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {samples_file.name}\n")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("✓ EDA COMPLETE")
    print("=" * 80)
    print(f"\nOutput folder: {out_dir.absolute()}\n")
    print(f"Files created:")
    print(f"  1. eda_qa_analysis.json ({len(qa_results)} items)")
    print(f"  2. eda_doc_analysis.json ({len(doc_results)} items)")
    print(f"  3. eda_patterns_summary.txt")
    print(f"  4. eda_chunking_comparison.json ({len(chunk_results)} items)")
    print(f"  5. eda_chunking_stats.txt")
    print(f"  6. eda_samples.json")
    print(f"\nKey findings:")
    print(f"  • QA with Điều: {qa_stats['with_dieu']} ({100*qa_stats['with_dieu']/total_qa:.1f}%)")
    print(f"  • QA without Điều: {qa_stats['without_dieu']} ({100*qa_stats['without_dieu']/total_qa:.1f}%)")
    if doc_stats["total_docs"] > 0:
        print(f"  • Docs with Điều: {doc_stats['with_dieu']} ({100*doc_stats['with_dieu']/doc_stats['total_docs']:.1f}%)")
        print(f"  • Docs without Điều: {doc_stats['without_dieu']} ({100*doc_stats['without_dieu']/doc_stats['total_docs']:.1f}%)")


if __name__ == "__main__":
    main()
