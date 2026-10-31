"""Self-check: run with  python test_agentdrift.py"""
from agentdrift import report
from agentdrift.agent import MockAgent
from agentdrift.runner import run_task, sweep
from agentdrift.tasks import TASKS


def main():
    # gold chain sanity: every task must be solvable
    from agentdrift.tasks import gold_chain
    for t in TASKS:
        g = gold_chain(t)
        assert g[-1] is not None, f"{t.id}: gold answer is None"
        print(f"  {t.id:18s} len={len(t.steps)}  gold={g[-1]:.4f}")

    # perfect agent never fails
    r = run_task(TASKS[2], MockAgent(p=0.0, seed=1))
    assert r.success and r.onset is None, "perfect agent failed a task"

    # drifting agent: onset exists, recovery below 100%
    m = sweep(MockAgent(p=0.1, seed=1), TASKS, repeats=30)
    assert m["success_rate"] < 1.0, "10%-error agent should fail sometimes"
    assert m["recovery_rate"] is not None and m["recovery_rate"] <= 1.0
    # calibration: independent errors -> amplification near 1 (estimator sanity)
    amps = [d["amplification"] for d in m["per_length"]]
    assert any(0.5 <= a <= 1.5 for a in amps), f"calibration amps look broken: {amps}"

    # error growth exists on numeric chains with onset
    assert m["median_error_growth"] is not None and m["median_error_growth"] >= 1.0, \
        f"median error growth should be >= 1, got {m['median_error_growth']}"

    # report renders and contains the numbers
    html_out = report.render(m, "self-check")
    assert "amplification" in html_out and "<svg" in html_out

    print("OK: gold chains, perfect agent, drift metrics, report render")
    print("longest-chain amplification:",
          m["per_length"][-1]["amplification"])


if __name__ == "__main__":
    main()