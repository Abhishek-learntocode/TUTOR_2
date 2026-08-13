import os
import sys
import json
import time
import statistics
from collections import defaultdict

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_dir = os.path.join(backend_dir, "evaluation", "results")

raw_outputs_path = os.path.join(results_dir, "phase5b2_raw_model_outputs.jsonl")
ledger_path = os.path.join(results_dir, "phase5b2_request_ledger.json")
judge_path = os.path.join(results_dir, "phase5b2_raw_judge_results.jsonl")
metadata_path = os.path.join(results_dir, "phase5b2_model_metadata.json")
manifest_path = os.path.join(results_dir, "phase5b2_frozen_context_manifest.json")

print("=" * 80)
print("PHASE 5B.2B — ZERO-API DATASET INTEGRITY AUDIT")
print("=" * 80)

# Check files existence
files_status = {
    "raw_outputs": os.path.exists(raw_outputs_path),
    "ledger": os.path.exists(ledger_path),
    "judge": os.path.exists(judge_path),
    "metadata": os.path.exists(metadata_path),
    "manifest": os.path.exists(manifest_path),
}
print(f"[*] Artifact Files Existence: {files_status}")

# AUDIT 1: Raw Generation Dataset
raw_records = []
if files_status["raw_outputs"]:
    with open(raw_outputs_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if line.strip():
                try:
                    obj = json.loads(line.strip())
                    obj["_line_number"] = idx
                    raw_records.append(obj)
                except Exception as e:
                    print(f"[!] Syntax error reading line {idx} of raw_model_outputs: {e}")

total_records = len(raw_records)
unique_query_ids = sorted(list(set(r.get("query_id") for r in raw_records if r.get("query_id"))))
unique_model_keys = sorted(list(set(r.get("model_key") for r in raw_records if r.get("model_key"))))
unique_requested_models = sorted(list(set(r.get("requested_model") for r in raw_records if r.get("requested_model"))))
unique_actual_models = sorted(list(set(r.get("actual_model") for r in raw_records if r.get("actual_model"))))

ollama_records = [r for r in raw_records if r.get("provider") == "ollama"]
openrouter_records = [r for r in raw_records if r.get("provider") == "openrouter"]
successful_openrouter_records = [r for r in openrouter_records if r.get("status_code") == 200]
failed_openrouter_records = [r for r in openrouter_records if r.get("status_code") != 200]

records_with_request_id = [r for r in openrouter_records if r.get("request_id") and r.get("request_id") != "N/A"]
req_ids = [r.get("request_id") for r in records_with_request_id]
unique_request_ids = sorted(list(set(req_ids)))
duplicate_request_ids = [rid for rid in unique_request_ids if req_ids.count(rid) > 1]

print(f"\n---> AUDIT 1 SUMMARY:")
print(f"  Total Raw Generation Records   : {total_records}")
print(f"  Unique Query IDs ({len(unique_query_ids)}) : {unique_query_ids}")
print(f"  Unique Model Keys              : {unique_model_keys}")
print(f"  Ollama Records                 : {len(ollama_records)}")
print(f"  OpenRouter Records             : {len(openrouter_records)} (Success: {len(successful_openrouter_records)}, Failed: {len(failed_openrouter_records)})")
print(f"  OpenRouter Unique Request IDs  : {len(unique_request_ids)} (Duplicates: {len(duplicate_request_ids)})")

# AUDIT 2: Expected Model Matrix
expected_models = ["OLLAMA_BASELINE", "OPENROUTER_MODEL_A", "OPENROUTER_MODEL_B"]
expected_query_ids = [
    "factual_001", "factual_002", "factual_003",
    "concept_001", "concept_002", "concept_003",
    "multihop_001", "multihop_002", "multihop_004",
    "doc_001", "doc_003",
    "compare_001", "multidoc_001",
    "exam_001", "missing_002"
]

matrix = {}
missing_combinations = []
duplicate_combinations = []
unexpected_combinations = []

query_model_counts = defaultdict(lambda: defaultdict(int))
for r in raw_records:
    qid = r.get("query_id")
    mkey = r.get("model_key")
    if qid and mkey:
        query_model_counts[qid][mkey] += 1

for qid in expected_query_ids:
    matrix[qid] = {}
    for mkey in expected_models:
        cnt = query_model_counts[qid][mkey]
        matrix[qid][mkey] = cnt
        if cnt == 0:
            missing_combinations.append((qid, mkey))
        elif cnt > 1:
            duplicate_combinations.append((qid, mkey, cnt))

for qid in query_model_counts:
    if qid not in expected_query_ids:
        for mkey, cnt in query_model_counts[qid].items():
            unexpected_combinations.append((qid, mkey, cnt))
    else:
        for mkey in query_model_counts[qid]:
            if mkey not in expected_models:
                unexpected_combinations.append((qid, mkey, query_model_counts[qid][mkey]))

print(f"\n---> AUDIT 2 SUMMARY (Model Matrix):")
print(f"  Theoretical Expected Records : {len(expected_query_ids)} x 3 = {len(expected_query_ids)*3}")
print(f"  Actual Matrix Records        : {total_records}")
print(f"  Missing Combinations ({len(missing_combinations)}) : {missing_combinations}")
print(f"  Duplicate Combinations ({len(duplicate_combinations)}) : {duplicate_combinations}")
print(f"  Unexpected Combinations ({len(unexpected_combinations)}) : {unexpected_combinations}")

# AUDIT 3: OpenRouter Request Count
actual_openrouter_requests = len(openrouter_records)
actual_remaining_budget = 40 - actual_openrouter_requests

print(f"\n---> AUDIT 3 SUMMARY (OpenRouter Budget):")
print(f"  OpenRouter Total Records      : {actual_openrouter_requests}")
print(f"  OpenRouter Unique Request IDs : {len(unique_request_ids)}")
print(f"  OpenRouter Successful         : {len(successful_openrouter_records)}")
print(f"  OpenRouter Failed             : {len(failed_openrouter_records)}")
print(f"  Hard Budget Limit             : 40")
print(f"  True Remaining Budget         : {actual_remaining_budget}")

# AUDIT 4: Request Ledger
ledger_data = {}
ledger_vs_raw_pass = True
ledger_discrepancies = []

if files_status["ledger"]:
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger_data = json.load(f)
    except Exception as e:
        ledger_vs_raw_pass = False
        ledger_discrepancies.append(f"Failed to read ledger JSON: {e}")

ledger_req_used = ledger_data.get("openrouter_requests_used", 0)
ledger_req_remaining = ledger_data.get("openrouter_requests_remaining", 0)
ledger_entries = ledger_data.get("ledger", [])

ledger_numbers = [entry.get("request_number") for entry in ledger_entries if entry.get("request_number") is not None]
dup_ledger_numbers = [n for n in set(ledger_numbers) if ledger_numbers.count(n) > 1]
gaps_in_numbering = []
if ledger_numbers:
    expected_nums = list(range(1, max(ledger_numbers) + 1))
    gaps_in_numbering = [n for n in expected_nums if n not in ledger_numbers]

ledger_req_ids = [entry.get("request_id") for entry in ledger_entries if entry.get("request_id") and entry.get("request_id") != "N/A"]
dup_ledger_req_ids = [rid for rid in set(ledger_req_ids) if ledger_req_ids.count(rid) > 1]

if ledger_req_used != actual_openrouter_requests:
    ledger_vs_raw_pass = False
    ledger_discrepancies.append(f"Ledger reports openrouter_requests_used={ledger_req_used}, but raw dataset contains {actual_openrouter_requests} OpenRouter records.")

if ledger_req_remaining != (40 - actual_openrouter_requests):
    ledger_vs_raw_pass = False
    ledger_discrepancies.append(f"Ledger reports openrouter_requests_remaining={ledger_req_remaining}, but actual remaining budget is {40 - actual_openrouter_requests}.")

if len(ledger_entries) != actual_openrouter_requests:
    ledger_vs_raw_pass = False
    ledger_discrepancies.append(f"Ledger list contains {len(ledger_entries)} entries, but raw dataset contains {actual_openrouter_records} OpenRouter records.")

print(f"\n---> AUDIT 4 SUMMARY (Ledger vs Raw):")
print(f"  Ledger Reported Used      : {ledger_req_used}")
print(f"  Ledger Reported Remaining : {ledger_req_remaining}")
print(f"  Ledger Entries Count      : {len(ledger_entries)}")
print(f"  Duplicate Ledger Numbers  : {dup_ledger_numbers}")
print(f"  Gaps in Request Numbers   : {gaps_in_numbering}")
print(f"  Duplicate Ledger Req IDs  : {dup_ledger_req_ids}")
print(f"  LEDGER_VS_RAW_STATUS     : {'PASS' if ledger_vs_raw_pass else 'FAIL'}")
if ledger_discrepancies:
    print(f"  Discrepancies             : {ledger_discrepancies}")

# AUDIT 5: Query Count
manifest_data = {}
if files_status["manifest"]:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception as e:
        print(f"[!] Error reading manifest: {e}")

manifest_query_ids = sorted(list(manifest_data.keys()))

judge_records = []
if files_status["judge"]:
    with open(judge_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if line.strip():
                try:
                    judge_records.append(json.loads(line.strip()))
                except Exception as e:
                    print(f"[!] Error reading line {idx} of judge results: {e}")

judge_query_ids = sorted(list(set(r.get("query_id") for r in judge_records if r.get("query_id"))))

true_query_count = len(unique_query_ids)
print(f"\n---> AUDIT 5 SUMMARY (Query Count):")
print(f"  Raw Dataset Unique Queries : {true_query_count}")
print(f"  Context Manifest Queries   : {len(manifest_query_ids)}")
print(f"  Judge Results Queries      : {len(judge_query_ids)}")
print(f"  Expected Queries           : {len(expected_query_ids)}")

# AUDIT 6: Frozen Prompt Control
prompt_control_pass_count = 0
prompt_control_fail_count = 0
prompt_fail_queries = []

query_prompt_hashes = defaultdict(set)
for r in raw_records:
    qid = r.get("query_id")
    phash = r.get("prompt_sha256")
    if qid and phash:
        query_prompt_hashes[qid].add(phash)

for qid in unique_query_ids:
    hashes = query_prompt_hashes[qid]
    if len(hashes) == 1:
        prompt_control_pass_count += 1
    else:
        prompt_control_fail_count += 1
        prompt_fail_queries.append((qid, list(hashes)))

prompt_control_pass = (prompt_control_fail_count == 0)
print(f"\n---> AUDIT 6 SUMMARY (Prompt Hash Control):")
print(f"  Prompt Hash Pass Count : {prompt_control_pass_count}")
print(f"  Prompt Hash Fail Count : {prompt_control_fail_count}")
print(f"  PROMPT_CONTROL_STATUS  : {'PASS' if prompt_control_pass else 'FAIL'}")

# AUDIT 7: Frozen Context Control
context_control_pass_count = 0
context_control_fail_count = 0
context_fail_queries = []

query_context_hashes = defaultdict(set)
for r in raw_records:
    qid = r.get("query_id")
    chash = r.get("context_sha256")
    if qid and chash:
        query_context_hashes[qid].add(chash)

for qid in unique_query_ids:
    hashes = query_context_hashes[qid]
    if len(hashes) == 1:
        context_control_pass_count += 1
    else:
        context_control_fail_count += 1
        context_fail_queries.append((qid, list(hashes)))

context_control_pass = (context_control_fail_count == 0)
print(f"\n---> AUDIT 7 SUMMARY (Context Hash Control):")
print(f"  Context Hash Pass Count : {context_control_pass_count}")
print(f"  Context Hash Fail Count : {context_control_fail_count}")
print(f"  CONTEXT_CONTROL_STATUS  : {'PASS' if context_control_pass else 'FAIL'}")

# AUDIT 8: Model Identity
model_identity_mismatches = []
for r in openrouter_records:
    req_m = r.get("requested_model")
    act_m = r.get("actual_model")
    if req_m and act_m and req_m != act_m:
        model_identity_mismatches.append({
            "query_id": r.get("query_id"),
            "request_id": r.get("request_id"),
            "requested": req_m,
            "actual": act_m,
        })

model_identity_pass = (len(model_identity_mismatches) == 0)
print(f"\n---> AUDIT 8 SUMMARY (Model Identity Control):")
print(f"  Mismatches Count       : {len(model_identity_mismatches)}")
print(f"  MODEL_IDENTITY_STATUS  : {'PASS' if model_identity_pass else 'FAIL'}")

# AUDIT 9: Judge Data & Cross-Reference
total_judge_records = len(judge_records)
judge_unique_queries = sorted(list(set(r.get("query_id") for r in judge_records if r.get("query_id"))))
judge_unique_models = sorted(list(set(r.get("model_key") for r in judge_records if r.get("model_key"))))

judge_combos = set((r.get("query_id"), r.get("model_key")) for r in judge_records)
raw_combos = set((r.get("query_id"), r.get("model_key")) for r in raw_records)

judge_without_generation = judge_combos - raw_combos
generation_without_judge = raw_combos - judge_combos

judge_reconciliation_pass = (len(judge_without_generation) == 0 and len(generation_without_judge) == 0)

print(f"\n---> AUDIT 9 SUMMARY (Judge Cross-Reference):")
print(f"  Total Judge Records          : {total_judge_records}")
print(f"  Judge Unique Queries         : {len(judge_unique_queries)}")
print(f"  Judge Unique Model Keys      : {judge_unique_models}")
print(f"  Judge Without Generation     : {len(judge_without_generation)}")
print(f"  Generation Without Judge     : {len(generation_without_judge)}")
print(f"  JUDGE_RECONCILIATION_STATUS  : {'PASS' if judge_reconciliation_pass else 'FAIL'}")

# AUDIT 10: Judge Score Sanity
malformed_scores = []
score_keys = ["correctness_score", "grounding_score", "completeness_score", "relevance_score", "instruction_following_score"]
for idx, r in enumerate(judge_records, 1):
    for skey in score_keys:
        val = r.get(skey)
        if val is None or not (0.0 <= float(val) <= 5.0):
            malformed_scores.append((idx, r.get("query_id"), r.get("model_key"), skey, val))

score_sanity_pass = (len(malformed_scores) == 0)
print(f"\n---> AUDIT 10 SUMMARY (Judge Score Sanity):")
print(f"  Malformed Scores Count       : {len(malformed_scores)}")
print(f"  SCORE_SANITY_STATUS          : {'PASS' if score_sanity_pass else 'FAIL'}")

# AUDIT 11: Category Distribution
category_queries = defaultdict(set)
for r in raw_records:
    cat = r.get("category")
    qid = r.get("query_id")
    if cat and qid:
        category_queries[cat].add(qid)

cat_distribution = {cat: len(qset) for cat, qset in category_queries.items()}
print(f"\n---> AUDIT 11 SUMMARY (Category Distribution):")
for cat, cnt in cat_distribution.items():
    print(f"  {cat:20s}: {cnt} queries ({sorted(list(category_queries[cat]))})")

# AUDIT 12: Latency / Token Sanity
missing_latency = [r for r in raw_records if r.get("generation_latency_sec") is None]
negative_latency = [r for r in raw_records if r.get("generation_latency_sec") is not None and r.get("generation_latency_sec") < 0]
missing_tokens = [r for r in raw_records if r.get("total_tokens") is None]
negative_tokens = [r for r in raw_records if r.get("total_tokens") is not None and r.get("total_tokens") < 0]

latencies_by_model = defaultdict(list)
tokens_by_model = defaultdict(list)

for r in raw_records:
    mkey = r.get("model_key")
    lat = r.get("generation_latency_sec")
    tok = r.get("total_tokens")
    if mkey and lat is not None:
        latencies_by_model[mkey].append(lat)
    if mkey and tok is not None:
        tokens_by_model[mkey].append(tok)

avg_latency_by_model = {m: round(statistics.mean(l), 4) if l else 0.0 for m, l in latencies_by_model.items()}
median_latency_by_model = {m: round(statistics.median(l), 4) if l else 0.0 for m, l in latencies_by_model.items()}
avg_tokens_by_model = {m: round(statistics.mean(t), 2) if t else 0.0 for m, t in tokens_by_model.items()}

print(f"\n---> AUDIT 12 SUMMARY (Descriptive Metrics):")
print(f"  Missing Latency Count   : {len(missing_latency)}")
print(f"  Negative Latency Count  : {len(negative_latency)}")
print(f"  Missing Tokens Count    : {len(missing_tokens)}")
print(f"  Negative Tokens Count   : {len(negative_tokens)}")
print(f"  Average Latency by Model: {avg_latency_by_model}")
print(f"  Median Latency by Model : {median_latency_by_model}")
print(f"  Average Tokens by Model : {avg_tokens_by_model}")

# OVERALL DATASET INTEGRITY VERDICT
integrity_issues = []

if not ledger_vs_raw_pass:
    integrity_issues.extend(ledger_discrepancies)

if not prompt_control_pass:
    integrity_issues.append(f"Prompt hash control failed for queries: {prompt_fail_queries}")

if not context_control_pass:
    integrity_issues.append(f"Context hash control failed for queries: {context_fail_queries}")

if not model_identity_pass:
    integrity_issues.append(f"Model identity mismatch found: {model_identity_mismatches}")

if not judge_reconciliation_pass:
    integrity_issues.append(f"Judge reconciliation failed (judge without gen: {len(judge_without_generation)}, gen without judge: {len(generation_without_judge)})")

if not score_sanity_pass:
    integrity_issues.append(f"Malformed judge scores found: {malformed_scores}")

if actual_openrouter_requests > 40:
    integrity_issues.append(f"OpenRouter hard budget exceeded: used {actual_openrouter_requests} > 40 Limit!")

overall_verdict = "PASS" if len(integrity_issues) == 0 else "FAIL"

print("\n==================================================================")
print(f"FINAL DATASET INTEGRITY VERDICT: DATASET_INTEGRITY: {overall_verdict}")
if integrity_issues:
    print(f"Discrepancies List:")
    for issue in integrity_issues:
        print(f" - {issue}")
print("==================================================================")

# OUTPUT 1: evaluation/results/phase5b2_dataset_integrity_audit.json
audit_json_path = os.path.join(results_dir, "phase5b2_dataset_integrity_audit.json")
audit_json_data = {
    "verdict": f"DATASET_INTEGRITY: {overall_verdict}",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "raw_record_count": total_records,
    "unique_query_count": true_query_count,
    "expected_query_count": len(expected_query_ids),
    "model_count": len(unique_model_keys),
    "ollama_generation_records": len(ollama_records),
    "openrouter_generation_records": len(openrouter_records),
    "openrouter_requests_true": actual_openrouter_requests,
    "openrouter_requests_ledger": ledger_req_used,
    "openrouter_request_ids_unique": len(unique_request_ids),
    "openrouter_budget": 40,
    "openrouter_remaining": actual_remaining_budget,
    "judge_record_count": total_judge_records,
    "prompt_control": "PASS" if prompt_control_pass else "FAIL",
    "context_control": "PASS" if context_control_pass else "FAIL",
    "model_identity_control": "PASS" if model_identity_pass else "FAIL",
    "ledger_reconciliation": "PASS" if ledger_vs_raw_pass else "FAIL",
    "judge_reconciliation": "PASS" if judge_reconciliation_pass else "FAIL",
    "score_sanity": "PASS" if score_sanity_pass else "FAIL",
    "duplicates_count": len(duplicate_combinations) + len(dup_ledger_req_ids),
    "missing_model_query_combinations": len(missing_combinations),
    "missing_combinations_detail": [f"{q}:{m}" for q, m in missing_combinations],
    "discrepancies": integrity_issues,
    "descriptive_metrics": {
        "category_distribution": cat_distribution,
        "avg_latency_sec": avg_latency_by_model,
        "median_latency_sec": median_latency_by_model,
        "avg_total_tokens": avg_tokens_by_model,
    }
}

with open(audit_json_path, "w", encoding="utf-8") as f:
    json.dump(audit_json_data, f, indent=2)

# OUTPUT 2: evaluation/results/phase5b2_dataset_integrity_audit.md
audit_md_path = os.path.join(results_dir, "phase5b2_dataset_integrity_audit.md")
md_lines = [
    "------------------------------------------------------------",
    "PHASE 5B.2B DATASET INTEGRITY AUDIT",
    "------------------------------------------------------------",
    "",
    f"RAW_RECORD_COUNT: {total_records}",
    f"UNIQUE_QUERY_COUNT: {true_query_count}",
    f"EXPECTED_QUERY_COUNT: {len(expected_query_ids)}",
    f"MODEL_COUNT: {len(unique_model_keys)}",
    "",
    f"OLLAMA_GENERATION_RECORDS: {len(ollama_records)}",
    f"OPENROUTER_GENERATION_RECORDS: {len(openrouter_records)}",
    "",
    f"OPENROUTER_REQUESTS_TRUE: {actual_openrouter_requests}",
    f"OPENROUTER_REQUESTS_LEDGER: {ledger_req_used}",
    f"OPENROUTER_REQUEST_IDS_UNIQUE: {len(unique_request_ids)}",
    "",
    f"OPENROUTER_BUDGET: 40",
    f"OPENROUTER_REMAINING: {actual_remaining_budget}",
    "",
    f"JUDGE_RECORD_COUNT: {total_judge_records}",
    "",
    f"PROMPT_CONTROL:",
    f"{'PASS' if prompt_control_pass else 'FAIL'}",
    "",
    f"CONTEXT_CONTROL:",
    f"{'PASS' if context_control_pass else 'FAIL'}",
    "",
    f"MODEL_IDENTITY_CONTROL:",
    f"{'PASS' if model_identity_pass else 'FAIL'}",
    "",
    f"LEDGER_RECONCILIATION:",
    f"{'PASS' if ledger_vs_raw_pass else 'FAIL'}",
    "",
    f"JUDGE_RECONCILIATION:",
    f"{'PASS' if judge_reconciliation_pass else 'FAIL'}",
    "",
    f"DUPLICATES:",
    f"{len(duplicate_combinations) + len(dup_ledger_req_ids)}",
    "",
    f"MISSING_MODEL_QUERY_COMBINATIONS:",
    f"{len(missing_combinations)}",
    "",
    "============================================================",
    "FINAL VERDICT",
    "============================================================",
    "",
    f"DATASET_INTEGRITY: {overall_verdict}",
]

if overall_verdict == "FAIL":
    md_lines.append("")
    md_lines.append("Discrepancies Found:")
    for issue in integrity_issues:
        md_lines.append(f" - {issue}")

with open(audit_md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\nSaved audit JSON to: {audit_json_path}")
print(f"Saved audit Markdown to: {audit_md_path}")
