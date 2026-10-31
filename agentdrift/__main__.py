"""CLI: python -m agentdrift demo | eval"""
import argparse
import sys

from . import report, tasks as T
from .agent import MockAgent, OpenAIAgent
from .runner import run_task, sweep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentdrift")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="deterministic simulation backend, no API key needed")
    d.add_argument("--p", type=float, default=0.08, help="simulated per-step error rate")
    d.add_argument("--repeats", type=int, default=25)
    d.add_argument("--out", default="report.html")

    e = sub.add_parser("eval", help="run a real OpenAI-compatible model")
    e.add_argument("--model", default=None)
    e.add_argument("--repeats", type=int, default=1)
    e.add_argument("--out", default="report.html")

    a = ap.parse_args(argv)
    if a.cmd == "demo":
        agent = MockAgent(p=a.p, seed=42)
        metrics = sweep(agent, T.TASKS, repeats=a.repeats)
        title = f"simulated agent, p={a.p}, {a.repeats} repeats/task"
    else:
        agent = OpenAIAgent(model=a.model)
        metrics = sweep(agent, T.TASKS, repeats=a.repeats)
        title = f"model={agent.model}"

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(report.render(metrics, title))
    print(f"wrote {a.out}")
    print(f"success={metrics['success_rate']:.0%}  "
          f"per-step error={metrics['per_step_error_rate']:.1%}  "
          f"recovery={metrics['recovery_rate']}")


if __name__ == "__main__":
    main()