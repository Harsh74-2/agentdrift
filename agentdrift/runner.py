"""Runner and drift metrics.

Metrics (the interview-ready definitions):
- success_rate: fraction of full chains ending in the correct final answer.
- onset step: the first step whose result diverges from the gold chain.
- recovery_rate: of chains with an onset error, the fraction that still reach
  the correct final answer (real agents self-correct; the bundled mock cannot).
- amplification: observed failure rate vs the independence baseline
  1-(1-p)^L, with p estimated from onset rates (fresh errors only).
  Amplification > 1 means errors interact; < 1 means the agent recovers.
- median_error_growth: |final error| / |first error| on numeric chains —
  how much a single mistake grows by the end of the chain. The direct
  analogue of rollout error accumulation in physical simulation.
"""
from dataclasses import dataclass, field
from .tasks import Task, call, gold_chain


@dataclass
class StepRecord:
    index: int
    goal: str
    action: dict
    result: object
    gold: object
    error: bool


@dataclass
class RunResult:
    task_id: str
    length: int
    success: bool
    onset: int | None          # first diverging step index
    recovered: bool            # onset error but still correct final answer
    steps: list = field(default_factory=list)


def run_task(task: Task, agent) -> RunResult:
    """Execute one chain. `agent.act(task_id, i, goal, history) -> action`."""
    if hasattr(agent, "known"):
        agent.known.setdefault(task.id, task.steps)
    gold = gold_chain(task)
    results, history, records = [], [], []
    onset = None
    for i, s in enumerate(task.steps):
        try:
            action = agent.act(task.id, i, s.goal, history)
            result = call(action["tool"], action["args"], results)
        except Exception as e:  # malformed action, bad tool, disallowed expr
            result = f"error: {type(e).__name__}"
            action = {"error": str(e)}
        error = not _equal(result, gold[i])
        if error and onset is None:
            onset = i
        records.append(StepRecord(i, s.goal, action, result, gold[i], error))
        results.append(result if not isinstance(result, str) or not result.startswith("error:") else None)
        history.append({"goal": s.goal, "action": action, "result": result})
    final_ok = _equal(results[-1] if results else None, gold[-1])
    return RunResult(task.id, len(task.steps), final_ok, onset,
                     onset is not None and final_ok, records)


def _equal(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def sweep(agent, tasks, repeats: int = 10) -> dict:
    """Run `repeats` trials per task; aggregate drift metrics."""
    runs = []
    for _ in range(repeats):
        runs += [run_task(t, agent) for t in tasks]
    by_len = {}
    for r in runs:
        by_len.setdefault(r.length, []).append(r)

    def rate(rs, pred):
        return sum(1 for r in rs if pred(r)) / len(rs)

    # Intrinsic per-step error rate: onset = first divergence, so it counts
    # only fresh errors, never ones cascaded from an earlier mistake.
    # E[onset per chain] = 1-(1-p)^L for mean length L -> solve for p.
    mean_L = max(1.0, sum(r.length for r in runs) / len(runs))
    onset_rate_all = rate(runs, lambda r: r.onset is not None)
    p_est = 1 - (1 - onset_rate_all) ** (1 / mean_L)

    per_len = {}
    for L, rs in sorted(by_len.items()):
        observed_fail = 1 - rate(rs, lambda r: r.success)
        baseline_fail = 1 - (1 - p_est) ** L
        per_len[L] = {"length": L,
                      "success": 1 - observed_fail,
                      "onset": rate(rs, lambda r: r.onset is not None),
                      "baseline_fail": baseline_fail,
                      "amplification": observed_fail / max(baseline_fail, 1e-9)}
    all_onsets = [r for r in runs if r.onset is not None]

    # Error growth: how large the FINAL error is relative to the FIRST error,
    # numeric chains only. This is the compounding number that survives even
    # when failure == onset (no recovery possible), and the direct analogue of
    # rollout error accumulation in physical simulation.
    growths = []
    for r in all_onsets:
        rec = next(s for s in r.steps if s.index == r.onset)
        final = next((s for s in r.steps if s.index == r.length - 1), None)
        try:
            first_err = abs(float(rec.result) - float(rec.gold))
            final_err = abs(float(final.result) - float(final.gold))
            if first_err > 1e-9:
                growths.append(final_err / first_err)
        except (TypeError, ValueError):
            pass

    return {
        "repeats": repeats,
        "per_step_error_rate": p_est,
        "success_rate": rate(runs, lambda r: r.success),
        "recovery_rate": (sum(1 for r in all_onsets if r.recovered)
                          / len(all_onsets)) if all_onsets else None,
        "onset_rate": onset_rate_all,
        "median_error_growth": (sorted(growths)[len(growths) // 2]
                                if growths else None),
        "onset_distribution": {r.onset: 1 for r in all_onsets},
        "per_length": list(per_len.values()),
    }