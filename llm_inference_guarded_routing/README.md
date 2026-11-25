# Optimizing Open-Source LLM Inference Latency and VRAM Headroom with Guarded Routing

## Overview
This project diagnoses and resolves hidden latency and GPU memory bottlenecks in open-source LLM inference without relying on heavy external servers. Using **Google Colab** (running a compact open-source instruct model directly via `transformers`) and **Langflow** (as a visual routing and guardrail layer), the workflow separates prefill first-token delay (`TTFT`) from decode token-to-token latency drift (`TPOT` drift), establishes empirical input/output sequence limits (`ISL`/`OSL`), and replaces an unguarded single-path flow with criticality- and pressure-aware routing.

## Folder Structure

### Starter Input Files
* `workload_prompts.csv` — 12 benchmark prompts spanning four workload shapes (`short_in_short_out`, `long_in_short_out`, `short_in_long_out`, `long_in_long_out`) and two criticality tiers (`standard`, `high_accuracy`).
* `baseline_inference_traces.jsonl` — 16 pre-logged baseline runs capturing coarse `total_latency_ms` and peak VRAM spikes under uncapped `max_tokens=4096` settings.
* `inference_diagnostics.ipynb` — Google Colab notebook for baseline inspection, per-token timing/VRAM instrumentation, empirical `ISL`/`OSL` plotting, and post-fix verification.
* `naive_inference_router.json` — Starter single-path Langflow routing flow with wide token limits and no criticality, length, or latency-drift guardrails.

### Generated Deliverable Files
* `guarded_inference_router.json` — Upgraded multi-branch Langflow flow enforcing criticality protection, empirical length cutoffs, and latency/memory drift guardrails.
* `optimized_inference_traces.jsonl` — Post-fix verification traces logging separated `TTFT`, `TPOT` start/end drift, KV-cache delta, and routing decisions.
* `isl_osl_diagnostics.png` — Distribution plots showing empirical 90th-percentile `ISL` and `OSL` cutoffs.
* `latency_drift_comparison.png` — Before-and-after peak GPU VRAM comparison chart.
* `inference_sizing_decision_brief.md` — Concise (~100-word) engineering brief summarizing sizing bounds, trade-offs, and next-step scaling triggers.

## Workflow Phases
1. **Inspect Baseline & Naive Flow:** Review `baseline_inference_traces.jsonl` and `naive_inference_router.json` to expose how total latency hides `TTFT` vs. decode drift and uncapped KV-cache growth.
2. **Instrument & Size in Colab:** Patch `inference_diagnostics.ipynb` to measure per-token timing and VRAM headroom, plot `ISL`/`OSL` distributions, and pick empirical sequence bounds.
3. **Re-Architect Routing in Langflow:** Upgrade `naive_inference_router.json` into `guarded_inference_router.json` with branches for task criticality, sequence length limits, and drift protection.
4. **Verify & Document:** Run contrasting prompts across the updated setup, save `optimized_inference_traces.jsonl` and diagnostic plots, and write `inference_sizing_decision_brief.md`.