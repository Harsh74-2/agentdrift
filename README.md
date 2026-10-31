# agentdrift

**Measure how single-step errors compound across agent trajectories.**

An LLM agent that fumbles one tool call usually doesn't just fail that step
— the wrong value flows into every later step that depends on it, and the
final answer fails at magnitudes the original error never had. agentdrift is
a small, dependency-free harness that quantifies exactly that: per-step error
rates, where errors first appear, whether agents recover from them, and how
large a mistake grows by the time it reaches the answer.

It comes out of my bachelor thesis on neural surrogates for gravitational
N-body simulation, where the central finding was that autoregressive rollout
compounds error (~5x vs single-step feedback, measured across MLP/LSTM/GNN
architectures). agentdrift ports that question to agentic ML: *what does one
bad step do to the trajectory?*

## Quick start

```bash
# zero-config demo: deterministic simulation backend, no API key needed
python -m agentdrift demo --out report.html

# self-check: gold chains, perfect agent, metric calibration
python test_agentdrift.py

# against any OpenAI-compatible model (OpenAI, Together, ollama, vLLM, ...)
# config via env: AGENTDRIFT_BASE_URL / AGENTDRIFT_API_KEY / AGENTDRIFT_MODEL
python -m agentdrift eval --model gpt-4o-mini --out report.html
```

No install step, no dependencies: Python 3.10+ standard library only.

## What it measures

| Metric | Definition | Why it matters |
|---|---|---|
| per-step error rate | intrinsic rate, estimated from **onset** events (first divergence from the gold chain), so cascaded errors never inflate it | the agent's raw reliability |
| onset distribution | which step the first divergence lands on | where chains break |
| recovery rate | of chains with an onset error, fraction still reaching the correct final answer | self-correction ability; the metric that separates real agents from dead chains |
| amplification | observed failure rate vs the independence baseline `1-(1-p)^L` per chain length | `> 1` = errors interact; `< 1` = agent recovers |
| median error growth | final error / first error on numeric chains | how much one mistake magnifies by the end — the direct analogue of rollout error accumulation in physical simulation |

The report is a standalone HTML file with a drift curve (observed success vs
independent-error baseline), an onset table, and metric cards.

## Design notes

- **Tasks are honest by construction.** Eight deterministic tool-calling
  chains (lookup + safe arithmetic) where later steps reference earlier
  results (`r0 * r1`). Gold chains are executable, so grading is exact — no
  LLM judge, no subjective scoring.
- **The simulation backend validates the estimator.** `MockAgent` fumbles
  with independent probability `p`, and the measured amplification comes out
  ≈ 1 at every chain length — the metrics do what they claim. Calibration
  before application.
- **Recovery is the real prize.** The bundled task library makes a single
  error unrecoverable, so the mock's recovery rate is ~0 and failure ≈
  onset. Point the harness at a real model and its recovery rate and
  amplification become the interesting numbers: real agents see tool results
  (and error strings) in context and attempt repair — that is exactly the
  behaviour this harness scores.

## First real-model result

`gpt-oss:120b` (cloud, temperature 0) on the bundled task set, 2 repeats per
task (16 chains, small n):

| chain length | success | onset rate | independent baseline fail | amplification |
|---|---|---|---|---|
| 1 | 100% | 0% | 13.4% | 0.00x |
| 3 | 67% | 33% | 35.0% | 0.95x |
| 5 | 50% | 50% | 51.3% | 0.97x |
| 6 | 25% | 75% | 57.8% | **1.30x** |

Headline numbers: **56% overall success at 13.4% per-step error, and recovery
rate 0.0** — not once in 16 chains did the model correct its first mistake;
its error rate alone predicts roughly 100% task failure by length 6 (observed
25% vs 42% chance baseline). With error-free step 1 and onset clustering on
lookups, the failure mode is concrete, not statistical noise.

Reproduce: `AGENTDRIFT_BASE_URL=http://localhost:11434/v1
AGENTDRIFT_MODEL=gpt-oss:120b-cloud python -m agentdrift eval --repeats 2` (or
any OpenAI-compatible endpoint, cloud or local).

## Status

v0.1.1: working harness, calibrated estimator, HTML + JSON reports,
OpenAI-compatible backend tested against a real reasoning model.

MIT licensed.