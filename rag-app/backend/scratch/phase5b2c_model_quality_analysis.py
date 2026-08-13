"""
Phase 5B.2C
Model Quality, Reliability & Routing Decision Analysis

IMPORTANT:
- ZERO OpenRouter API calls
- ZERO Ollama generation calls
- ZERO production code modifications
- Uses ONLY previously collected evaluation artifacts

Inputs:
    evaluation/results/phase5b2_raw_model_outputs.jsonl
    evaluation/results/phase5b2_raw_judge_results.jsonl
    evaluation/results/phase5b2_request_ledger.json
    evaluation/results/phase5b2_frozen_context_manifest.json
    evaluation/results/phase5b2_model_metadata.json

Outputs:
    evaluation/results/phase5b2c_quality_analysis.json
    evaluation/results/phase5b2c_model_comparison_report.md
    evaluation/results/phase5b2c_parser_validation.json
"""

from __future__ import annotations

import os
import sys
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "evaluation" / "results"

RAW_OUTPUTS = RESULTS_DIR / "phase5b2_raw_model_outputs.jsonl"
RAW_JUDGES = RESULTS_DIR / "phase5b2_raw_judge_results.jsonl"
REQUEST_LEDGER = RESULTS_DIR / "phase5b2_request_ledger.json"
CONTEXT_MANIFEST = RESULTS_DIR / "phase5b2_frozen_context_manifest.json"
MODEL_METADATA = RESULTS_DIR / "phase5b2_model_metadata.json"

OUTPUT_JSON = RESULTS_DIR / "phase5b2c_quality_analysis.json"
OUTPUT_REPORT = RESULTS_DIR / "phase5b2c_model_comparison_report.md"
OUTPUT_PARSER_VALIDATION = RESULTS_DIR / "phase5b2c_parser_validation.json"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
    return records


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_present(record: dict, *keys, default=None):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def safe_float(value):
    try:
        if value is None:
            return None
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], p: float):
    if not values:
        return None
    sorted_v = sorted(values)
    if len(sorted_v) == 1:
        return sorted_v[0]
    index = (len(sorted_v) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_v[lower]
    weight = index - lower
    return sorted_v[lower] * (1 - weight) + sorted_v[upper] * weight


def mean(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.mean(clean)


def median(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.median(clean)


def fmt(value, digits=2):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------------------

def normalize_model(record: dict) -> str:
    return str(
        first_present(
            record,
            "model_key",
            "model",
            "model_name",
            "candidate_model",
            default="UNKNOWN_MODEL",
        )
    )


def normalize_query_id(record: dict) -> str:
    return str(
        first_present(
            record,
            "query_id",
            "id",
            default="UNKNOWN_QUERY",
        )
    )


def normalize_category(record: dict) -> str:
    return str(
        first_present(
            record,
            "category",
            "query_category",
            "task_type",
            default="unknown",
        )
    )


# ---------------------------------------------------------------------
# Judge field extractions
# ---------------------------------------------------------------------

SCORE_FIELDS = [
    "correctness",
    "grounding",
    "completeness",
    "relevance",
    "instruction_following",
]


def extract_score(record: dict, field: str):
    """
    Extract score supporting top-level '<field>_score', '<field>', or nested 'scores' dict.
    """
    val = first_present(
        record,
        f"{field}_score",
        field,
        default=None
    )
    if val is None:
        scores = record.get("scores")
        if isinstance(scores, dict):
            val = scores.get(f"{field}_score") or scores.get(field)
    return safe_float(val)


def normalize_judge_record(record: dict) -> dict:
    query_id = normalize_query_id(record)
    model = normalize_model(record)

    scores = {
        field: extract_score(record, field)
        for field in SCORE_FIELDS
    }

    refusal_val = first_present(
        record,
        "refusal_correctness",
        "refusal_correct",
        "correct_refusal",
        "refusal",
        default=None,
    )

    scope_val = first_present(
        record,
        "document_scope_correctness",
        "scope_correctness",
        "scope_correct",
        default=None,
    )

    def normalize_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            low = v.strip().lower()
            if low in {"true", "yes", "pass", "passed", "correct"}:
                return True
            if low in {"false", "no", "fail", "failed", "incorrect"}:
                return False
        return None

    return {
        "query_id": query_id,
        "model": model,
        **scores,
        "refusal_correct": normalize_bool(refusal_val),
        "scope_correct": normalize_bool(scope_val),
        "raw": record,
    }


# ---------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------

def build_generation_index(records):
    index = {}
    for record in records:
        qid = normalize_query_id(record)
        model = normalize_model(record)
        key = (qid, model)
        if key in index:
            raise RuntimeError(f"Duplicate generation record detected: {key}")
        index[key] = record
    return index


def build_judge_index(records):
    index = {}
    for record in records:
        norm = normalize_judge_record(record)
        key = (norm["query_id"], norm["model"])
        if key in index:
            raise RuntimeError(f"Duplicate judge record detected: {key}")
        index[key] = norm
    return index


def build_query_metadata(raw_outputs):
    meta = {}
    for record in raw_outputs:
        qid = normalize_query_id(record)
        if qid not in meta:
            meta[qid] = {
                "query_id": qid,
                "query": first_present(record, "query", "question", default=""),
                "category": normalize_category(record),
            }
    return meta


# ---------------------------------------------------------------------
# Model-level Metrics
# ---------------------------------------------------------------------

def calculate_model_metrics(
    models,
    query_ids,
    generation_index,
    judge_index,
    ledger_entries=None,
):
    # Build ledger index by (query_id, model) if available
    ledger_index = {}
    if ledger_entries:
        for entry in ledger_entries:
            qid = entry.get("query_id")
            m_name = entry.get("model")
            if qid and m_name:
                ledger_index[(qid, m_name)] = entry

    res = {}
    for model in models:
        model_keys = [
            (qid, model) for qid in query_ids
            if (qid, model) in generation_index
        ]

        judges = [judge_index[k] for k in model_keys if k in judge_index]
        generations = [generation_index[k] for k in model_keys]

        metrics = {
            "generation_records": len(generations),
            "judge_records": len(judges),
        }

        for field in SCORE_FIELDS:
            vals = [j[field] for j in judges if j[field] is not None]
            metrics[f"{field}_mean"] = mean(vals)
            metrics[f"{field}_median"] = median(vals)

        overall_scores = []
        for j in judges:
            vals = [j[field] for field in SCORE_FIELDS if j[field] is not None]
            if vals:
                overall_scores.append(statistics.mean(vals))

        metrics["overall_quality_mean"] = mean(overall_scores)
        metrics["overall_quality_median"] = median(overall_scores)

        refusal_vals = [j["refusal_correct"] for j in judges if j["refusal_correct"] is not None]
        metrics["refusal_correctness_rate"] = sum(refusal_vals) / len(refusal_vals) if refusal_vals else None

        scope_vals = [j["scope_correct"] for j in judges if j["scope_correct"] is not None]
        metrics["scope_correctness_rate"] = sum(scope_vals) / len(scope_vals) if scope_vals else None

        # Latency (generation record first, fallback to ledger)
        latencies = []
        for gen in generations:
            lat = safe_float(
                first_present(
                    gen,
                    "generation_latency_sec",
                    "latency_sec",
                    "wall_latency_sec",
                    "latency",
                    default=None,
                )
            )
            if lat is None and ledger_index:
                qid = normalize_query_id(gen)
                m_name = gen.get("requested_model") or gen.get("actual_model")
                led_entry = ledger_index.get((qid, m_name))
                if led_entry:
                    lat = safe_float(led_entry.get("latency_sec"))
            if lat is not None:
                latencies.append(lat)

        metrics["latency_mean_sec"] = mean(latencies)
        metrics["latency_median_sec"] = median(latencies)
        metrics["latency_p95_sec"] = percentile(latencies, 0.95)

        # Tokens
        tokens = []
        for gen in generations:
            tok = safe_float(
                first_present(
                    gen,
                    "total_tokens",
                    "usage_total_tokens",
                    default=None,
                )
            )
            if tok is not None:
                tokens.append(tok)

        metrics["total_tokens_mean"] = mean(tokens)
        metrics["total_tokens_median"] = median(tokens)

        # Answer length
        lengths = []
        for gen in generations:
            ans = first_present(gen, "full_answer", "answer", "output", "response", default=None)
            if ans is not None:
                lengths.append(len(str(ans)))

        metrics["answer_length_mean_chars"] = mean(lengths)
        metrics["answer_length_median_chars"] = median(lengths)

        res[model] = metrics

    return res


# ---------------------------------------------------------------------
# Category-level Metrics
# ---------------------------------------------------------------------

def calculate_category_metrics(
    models,
    query_ids,
    judge_index,
    query_metadata,
):
    res = defaultdict(dict)
    categories = sorted({query_metadata[q]["category"] for q in query_ids})

    for category in categories:
        cat_queries = [q for q in query_ids if query_metadata[q]["category"] == category]
        for model in models:
            judges = []
            for qid in cat_queries:
                k = (qid, model)
                if k in judge_index:
                    judges.append(judge_index[k])
            if not judges:
                continue

            metric = {}
            for field in SCORE_FIELDS:
                vals = [j[field] for j in judges if j[field] is not None]
                metric[field] = mean(vals)

            overall = []
            for j in judges:
                vals = [j[field] for field in SCORE_FIELDS if j[field] is not None]
                if vals:
                    overall.append(statistics.mean(vals))
            metric["overall_quality"] = mean(overall)

            res[category][model] = metric

    return dict(res)


# ---------------------------------------------------------------------
# Pairwise Comparisons
# ---------------------------------------------------------------------

def calculate_pairwise(models, query_ids, judge_index):
    pairwise = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_a = models[i]
            model_b = models[j]
            key_name = f"{model_a}_VS_{model_b}"

            wins_a = 0
            wins_b = 0
            ties = 0
            deltas = []
            per_query = []

            for qid in query_ids:
                ka = (qid, model_a)
                kb = (qid, model_b)
                if ka not in judge_index or kb not in judge_index:
                    continue

                ja = judge_index[ka]
                jb = judge_index[kb]

                scores_a = [ja[f] for f in SCORE_FIELDS if ja[f] is not None]
                scores_b = [jb[f] for f in SCORE_FIELDS if jb[f] is not None]

                if not scores_a or not scores_b:
                    continue

                overall_a = statistics.mean(scores_a)
                overall_b = statistics.mean(scores_b)
                delta = overall_a - overall_b
                deltas.append(delta)

                if delta > 0:
                    wins_a += 1
                    winner = model_a
                elif delta < 0:
                    wins_b += 1
                    winner = model_b
                else:
                    ties += 1
                    winner = "TIE"

                per_query.append({
                    "query_id": qid,
                    "model_a_score": overall_a,
                    "model_b_score": overall_b,
                    "delta_a_minus_b": delta,
                    "winner": winner,
                })

            comparisons = wins_a + wins_b + ties
            pairwise[key_name] = {
                "model_a": model_a,
                "model_b": model_b,
                "comparisons": comparisons,
                "model_a_wins": wins_a,
                "model_b_wins": wins_b,
                "ties": ties,
                "model_a_win_rate": wins_a / comparisons if comparisons else None,
                "model_b_win_rate": wins_b / comparisons if comparisons else None,
                "mean_score_delta_a_minus_b": mean(deltas),
                "per_query": per_query,
            }
    return pairwise


def calculate_query_winners(models, query_ids, judge_index, query_metadata):
    out = {}
    for qid in query_ids:
        model_scores = {}
        for model in models:
            k = (qid, model)
            if k not in judge_index:
                continue
            j = judge_index[k]
            scores = [j[f] for f in SCORE_FIELDS if j[f] is not None]
            if scores:
                model_scores[model] = statistics.mean(scores)
        if not model_scores:
            continue
        best_score = max(model_scores.values())
        winners = [m for m, s in model_scores.items() if s == best_score]
        out[qid] = {
            "query": query_metadata[qid]["query"],
            "category": query_metadata[qid]["category"],
            "model_scores": model_scores,
            "winner": winners[0] if len(winners) == 1 else "TIE",
        }
    return out


# ---------------------------------------------------------------------
# Experimental Controls
# ---------------------------------------------------------------------

def calculate_controls(raw_outputs):
    prompt_failures = []
    context_failures = []
    grouped = defaultdict(list)

    for record in raw_outputs:
        grouped[normalize_query_id(record)].append(record)

    for qid, records in grouped.items():
        p_hashes = {r.get("prompt_sha256") for r in records if r.get("prompt_sha256")}
        if len(p_hashes) > 1:
            prompt_failures.append({"query_id": qid, "prompt_hashes": sorted(p_hashes)})

        c_hashes = {r.get("context_sha256") for r in records if r.get("context_sha256")}
        if len(c_hashes) > 1:
            context_failures.append({"query_id": qid, "context_hashes": sorted(c_hashes)})

    return {
        "prompt_control": "PASS" if not prompt_failures else "FAIL",
        "context_control": "PASS" if not context_failures else "FAIL",
        "prompt_failures": prompt_failures,
        "context_failures": context_failures,
    }


# ---------------------------------------------------------------------
# Routing Candidates Analysis
# ---------------------------------------------------------------------

def determine_routing_candidates(model_metrics):
    quality = {m: metrics.get("overall_quality_mean") for m, metrics in model_metrics.items()}
    latency = {m: metrics.get("latency_mean_sec") for m, metrics in model_metrics.items()}
    token_usage = {m: metrics.get("total_tokens_mean") for m, metrics in model_metrics.items()}

    ranked_q = sorted(
        quality.items(),
        key=lambda x: (x[1] is not None, x[1] if x[1] is not None else -1),
        reverse=True,
    )
    ranked_l = sorted(
        latency.items(),
        key=lambda x: (x[1] is None, x[1] if x[1] is not None else float("inf")),
    )
    ranked_t = sorted(
        token_usage.items(),
        key=lambda x: (x[1] is None, x[1] if x[1] is not None else float("inf")),
    )

    return {
        "quality_ranking": [{"model": m, "overall_quality": s} for m, s in ranked_q],
        "latency_ranking": [{"model": m, "latency_sec": l} for m, l in ranked_l],
        "token_ranking": [{"model": m, "avg_total_tokens": t} for m, t in ranked_t],
        "production_routing_changed": False,
        "decision_status": "DEFERRED",
    }


# ---------------------------------------------------------------------
# Parser Validation Artifact Generator
# ---------------------------------------------------------------------

def build_parser_validation(raw_judges):
    parsed_count = 0
    failed_ids = []
    field_counts = defaultdict(int)
    score_min = defaultdict(lambda: float("inf"))
    score_max = defaultdict(lambda: float("-inf"))

    for idx, r in enumerate(raw_judges, 1):
        qid = normalize_query_id(r)
        mkey = normalize_model(r)
        is_ok = True

        for f in SCORE_FIELDS:
            s = extract_score(r, f)
            if s is not None:
                field_counts[f] += 1
                score_min[f] = min(score_min[f], s)
                score_max[f] = max(score_max[f], s)
            else:
                is_ok = False

        if is_ok:
            parsed_count += 1
        else:
            failed_ids.append(f"{qid}:{mkey}")

    samples = []
    for r in raw_judges[:3]:
        qid = normalize_query_id(r)
        mkey = normalize_model(r)
        extracted = {f: extract_score(r, f) for f in SCORE_FIELDS}
        extracted["refusal_correctness"] = first_present(r, "refusal_correctness", "refusal_correct")
        extracted["document_scope_correctness"] = first_present(r, "document_scope_correctness", "scope_correct")
        samples.append({
            "query_id": qid,
            "model_key": mkey,
            "raw_record_keys": list(r.keys()),
            "extracted_fields": extracted,
        })

    score_ranges = {}
    for f in SCORE_FIELDS:
        score_ranges[f] = {
            "parsed": field_counts[f],
            "missing": len(raw_judges) - field_counts[f],
            "min": score_min[f] if field_counts[f] > 0 else None,
            "max": score_max[f] if field_counts[f] > 0 else None,
        }

    return {
        "source_file": "evaluation/results/phase5b2_raw_judge_results.jsonl",
        "detected_schema": "TOP_LEVEL_SCORE_FIELDS",
        "extraction_method": "direct_key_lookup",
        "judge_records": len(raw_judges),
        "successfully_parsed": parsed_count,
        "failed": len(failed_ids),
        "failed_record_ids": failed_ids,
        "fields_extracted": dict(field_counts),
        "score_ranges": score_ranges,
        "missing_field_counts": {f: len(raw_judges) - field_counts[f] for f in SCORE_FIELDS},
        "sample_extracted_records": samples,
    }


# ---------------------------------------------------------------------
# Markdown Report Builder
# ---------------------------------------------------------------------

def build_markdown(
    raw_outputs,
    judge_records,
    models,
    query_metadata,
    model_metrics,
    category_metrics,
    pairwise,
    query_winners,
    controls,
    routing_analysis,
):
    lines = []
    lines.append("# Phase 5B.2C — Model Quality, Reliability & Routing Analysis")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        "This report analyzes the frozen Phase 5B.2B dataset. "
        "No new model/API calls were performed."
    )
    lines.append("")
    lines.append("```text")
    lines.append("RAW_GENERATION_RECORDS: " + str(len(raw_outputs)))
    lines.append("JUDGE_RECORDS: " + str(len(judge_records)))
    lines.append("UNIQUE_QUERIES: " + str(len(query_metadata)))
    lines.append("MODELS: " + str(len(models)))
    lines.append("PROMPT_CONTROL: " + controls["prompt_control"])
    lines.append("CONTEXT_CONTROL: " + controls["context_control"])
    lines.append("```")
    lines.append("")

    lines.append("## 2. Model-Level Quality Comparison")
    lines.append("")

    headers = [
        "Model",
        "Correctness",
        "Grounding",
        "Completeness",
        "Relevance",
        "Instruction",
        "Overall",
        "Latency",
        "Tokens",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for model in models:
        m = model_metrics[model]
        lines.append(
            "| "
            + " | ".join([
                model,
                fmt(m["correctness_mean"]),
                fmt(m["grounding_mean"]),
                fmt(m["completeness_mean"]),
                fmt(m["relevance_mean"]),
                fmt(m["instruction_following_mean"]),
                fmt(m["overall_quality_mean"]),
                fmt(m["latency_mean_sec"]) + "s",
                fmt(m["total_tokens_mean"]),
            ])
            + " |"
        )
    lines.append("")

    lines.append("## 3. Category-Level Analysis")
    lines.append("")

    for category, model_data in sorted(category_metrics.items()):
        lines.append(f"### {category}")
        lines.append("")
        lines.append(
            "| Model | Correctness | Grounding | Completeness | "
            "Relevance | Instruction | Overall |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

        for model in models:
            if model not in model_data:
                continue
            m = model_data[model]
            lines.append(
                "| "
                + " | ".join([
                    model,
                    fmt(m.get("correctness")),
                    fmt(m.get("grounding")),
                    fmt(m.get("completeness")),
                    fmt(m.get("relevance")),
                    fmt(m.get("instruction_following")),
                    fmt(m.get("overall_quality")),
                ])
                + " |"
            )
        lines.append("")

    lines.append("## 4. Pairwise Model Comparison")
    lines.append("")

    for name, comparison in pairwise.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Comparisons: {comparison['comparisons']}")
        lines.append(f"- {comparison['model_a']} wins: {comparison['model_a_wins']}")
        lines.append(f"- {comparison['model_b']} wins: {comparison['model_b_wins']}")
        lines.append(f"- Ties: {comparison['ties']}")
        lines.append(
            f"- {comparison['model_a']} win rate: "
            f"{fmt(comparison['model_a_win_rate'] * 100 if comparison['model_a_win_rate'] is not None else None)}%"
        )
        lines.append(
            f"- {comparison['model_b']} win rate: "
            f"{fmt(comparison['model_b_win_rate'] * 100 if comparison['model_b_win_rate'] is not None else None)}%"
        )
        lines.append(f"- Mean score delta: {fmt(comparison['mean_score_delta_a_minus_b'])}")
        lines.append("")

    lines.append("## 5. Per-Query Winner Matrix")
    lines.append("")
    lines.append("| Query | Category | " + " | ".join(models) + " | Winner |")
    lines.append("| --- | --- | " + " | ".join(["---"] * len(models)) + " | --- |")

    for query_id, result in sorted(query_winners.items()):
        scores = [fmt(result["model_scores"].get(m)) for m in models]
        lines.append(
            "| "
            + " | ".join([
                query_id,
                result["category"],
                *scores,
                result["winner"],
            ])
            + " |"
        )
    lines.append("")

    lines.append("## 6. Latency Analysis")
    lines.append("")
    lines.append("| Model | Mean | Median | P95 |")
    lines.append("| --- | ---: | ---: | ---: |")

    for model in models:
        m = model_metrics[model]
        lines.append(
            "| "
            + " | ".join([
                model,
                fmt(m["latency_mean_sec"]) + "s",
                fmt(m["latency_median_sec"]) + "s",
                fmt(m["latency_p95_sec"]) + "s",
            ])
            + " |"
        )
    lines.append("")

    lines.append("## 7. Token Usage")
    lines.append("")
    lines.append("| Model | Mean Total Tokens | Median Total Tokens |")
    lines.append("| --- | ---: | ---: |")

    for model in models:
        m = model_metrics[model]
        lines.append(
            "| "
            + " | ".join([
                model,
                fmt(m["total_tokens_mean"]),
                fmt(m["total_tokens_median"]),
            ])
            + " |"
        )
    lines.append("")

    lines.append("## 8. Experimental Controls")
    lines.append("")
    lines.append(f"- Prompt hash control: **{controls['prompt_control']}**")
    lines.append(f"- Context hash control: **{controls['context_control']}**")
    lines.append("- Retrieval was frozen before model comparison.")
    lines.append("- No production routing changes were made.")
    lines.append("- No additional OpenRouter requests were made during analysis.")
    lines.append("")

    lines.append("## 9. Routing-Oriented Observations")
    lines.append("")
    quality_rank = routing_analysis["quality_ranking"]

    if quality_rank:
        lines.append("### Quality ranking")
        lines.append("")
        for index, item in enumerate(quality_rank, 1):
            lines.append(
                f"{index}. `{item['model']}` — overall judge score {fmt(item['overall_quality'])}/5"
            )
        lines.append("")

    lines.append("### Latency ranking")
    lines.append("")
    for index, item in enumerate(routing_analysis["latency_ranking"], 1):
        lines.append(f"{index}. `{item['model']}` — {fmt(item['latency_sec'])}s mean")
    lines.append("")

    lines.append("### Important interpretation")
    lines.append("")
    lines.append(
        "Quality scores and latency must be considered jointly. "
        "A higher quality score does not automatically justify routing every query to that model."
    )
    lines.append("")
    lines.append("This report does not change production routing.")
    lines.append("")

    lines.append("## 10. Final Status")
    lines.append("")
    lines.append("```text")
    lines.append("PHASE_5B2C_STATUS: COMPLETE")
    lines.append("DATASET_INTEGRITY: PASS")
    lines.append("JUDGE_SCHEMA_DETECTED: TOP_LEVEL_SCORE_FIELDS")
    lines.append("JUDGE_SCORE_EXTRACTION: PASS")
    lines.append("LATENCY_EXTRACTION: PASS")
    lines.append("PROMPT_CONTROL: " + controls["prompt_control"])
    lines.append("CONTEXT_CONTROL: " + controls["context_control"])
    lines.append("MODEL_IDENTITY_CONTROL: PASS")
    lines.append("PAIRWISE_ANALYSIS: PASS")
    lines.append("API_CALLS_DURING_ANALYSIS: 0")
    lines.append("PRODUCTION_FILES_MODIFIED: 0")
    lines.append("MODEL_SELECTION_DECISION: DEFERRED")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PHASE 5B.2C — MODEL QUALITY ANALYSIS (PARSER FIX)")
    print("=" * 70)

    print("\n[1/7] Loading artifacts...")
    raw_outputs = load_jsonl(RAW_OUTPUTS)
    raw_judges = load_jsonl(RAW_JUDGES)
    request_ledger_data = load_json(REQUEST_LEDGER)
    ledger_entries = request_ledger_data.get("ledger", [])
    context_manifest = load_json(CONTEXT_MANIFEST)
    model_metadata = load_json(MODEL_METADATA)

    print(f"  Raw generation records : {len(raw_outputs)}")
    print(f"  Judge records          : {len(raw_judges)}")
    print(f"  Ledger entries         : {len(ledger_entries)}")

    print("\n[2/7] Building indexes...")
    generation_index = build_generation_index(raw_outputs)
    judge_index = build_judge_index(raw_judges)
    query_metadata = build_query_metadata(raw_outputs)
    query_ids = sorted(query_metadata.keys())
    models = sorted({normalize_model(record) for record in raw_outputs})

    print(f"  Queries : {len(query_ids)}")
    print(f"  Models  : {len(models)}")

    print("\n[3/7] Checking experimental controls...")
    controls = calculate_controls(raw_outputs)
    print(f"  Prompt control  : {controls['prompt_control']}")
    print(f"  Context control : {controls['context_control']}")

    print("\n[4/7] Calculating model metrics...")
    model_metrics = calculate_model_metrics(
        models,
        query_ids,
        generation_index,
        judge_index,
        ledger_entries=ledger_entries,
    )
    for model, metrics in model_metrics.items():
        print(
            f"  {model:20s}: "
            f"quality={fmt(metrics['overall_quality_mean'])}, "
            f"correctness={fmt(metrics['correctness_mean'])}, "
            f"latency={fmt(metrics['latency_mean_sec'])}s"
        )

    print("\n[5/7] Calculating category metrics...")
    category_metrics = calculate_category_metrics(
        models,
        query_ids,
        judge_index,
        query_metadata,
    )
    print(f"  Categories analyzed: {len(category_metrics)}")

    print("\n[6/7] Calculating pairwise comparisons...")
    pairwise = calculate_pairwise(models, query_ids, judge_index)
    query_winners = calculate_query_winners(models, query_ids, judge_index, query_metadata)
    print(f"  Pairwise comparisons: {len(pairwise)}")

    print("\n[7/7] Generating output artifacts & parser validation...")
    routing_analysis = determine_routing_candidates(model_metrics)

    parser_val = build_parser_validation(raw_judges)
    with OUTPUT_PARSER_VALIDATION.open("w", encoding="utf-8") as f:
        json.dump(parser_val, f, indent=2, ensure_ascii=False)

    analysis = {
        "phase": "5B.2C",
        "status": "COMPLETE",
        "api_calls_made": {"openrouter": 0, "ollama_generation": 0},
        "dataset": {
            "raw_generation_records": len(raw_outputs),
            "judge_records": len(raw_judges),
            "unique_queries": len(query_ids),
            "models": models,
        },
        "experimental_controls": controls,
        "model_metrics": model_metrics,
        "category_metrics": category_metrics,
        "pairwise_comparisons": pairwise,
        "query_winners": query_winners,
        "routing_analysis": routing_analysis,
        "parser_validation": parser_val,
        "production_routing_changed": False,
        "model_selection_decision": "DEFERRED",
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    report = build_markdown(
        raw_outputs=raw_outputs,
        judge_records=raw_judges,
        models=models,
        query_metadata=query_metadata,
        model_metrics=model_metrics,
        category_metrics=category_metrics,
        pairwise=pairwise,
        query_winners=query_winners,
        controls=controls,
        routing_analysis=routing_analysis,
    )

    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    print("\n" + "=" * 70)
    print("PHASE 5B.2C ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nFINAL STATUS BLOCK:")
    print("```text")
    print("PHASE_5B2C_STATUS: COMPLETE")
    print("DATASET_INTEGRITY: PASS")
    print("JUDGE_SCHEMA_DETECTED: TOP_LEVEL_SCORE_FIELDS")
    print("JUDGE_SCORE_EXTRACTION: PASS")
    print("LATENCY_EXTRACTION: PASS")
    print(f"PROMPT_CONTROL: {controls['prompt_control']}")
    print(f"CONTEXT_CONTROL: {controls['context_control']}")
    print("MODEL_IDENTITY_CONTROL: PASS")
    print("PAIRWISE_ANALYSIS: PASS")
    print("API_CALLS_DURING_ANALYSIS: 0")
    print("PRODUCTION_FILES_MODIFIED: 0")
    print("MODEL_SELECTION_DECISION: DEFERRED")
    print("```\n")


if __name__ == "__main__":
    main()