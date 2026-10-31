"""Task library: deterministic tool-calling tasks where step outputs feed later steps.

Two tools keep the surface honest and small:
  calc(expression)  -> float   (safe arithmetic, +-*/ and parentheses)
  lookup(key)       -> str|None (small in-process key-value store)

A Task is a chain of steps; a calc expression may reference the result of an
earlier step as r<i> (e.g. "r0 * r2"). The gold chain is executed with gold
args; the agent's chain is executed with the agent's actions. Any divergence
in an early step can cascade into the final answer.
"""
from dataclasses import dataclass
import ast
import operator
import re

STORE = {
    "price_apple": "1.20", "price_bread": "2.00", "price_cheese": "4.50",
    "tax_rate_food": "0.07", "population_lisbon": "545000",
    "population_porto": "231000", "area_lisbon_km2": "100.05",
    "area_porto_km2": "41.42", "moon_distance_km": "384400",
    "light_speed_kms": "299792", "basket_count": "3",
}


@dataclass
class Step:
    goal: str          # shown to the agent
    tool: str
    args: dict         # "expression" may reference prior results as r<i>


@dataclass
class Task:
    id: str
    description: str
    steps: list


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv}


def _safe_calc(expression: str) -> float:
    """Arithmetic-only expression evaluator (numbers, +-*/, parentheses)."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * ev(node.operand)
        raise ValueError(f"disallowed expression: {expression!r}")
    return float(ev(ast.parse(expression.strip(), mode="eval").body))


def call(tool: str, args: dict, results: list):
    """Execute a tool call. `results` holds results of prior steps in this chain."""
    if tool == "calc":
        substituted = re.sub(r"\br(\d+)\b",
                             lambda m: repr(results[int(m.group(1))]),
                             str(args["expression"]))
        return _safe_calc(substituted)
    if tool == "lookup":
        v = STORE.get(str(args["key"]))
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return v
    raise ValueError(f"unknown tool {tool!r}")


T = Task
TASKS = [
    T("one_lookup", "Price of apples?", [Step("apple price", "lookup", {"key": "price_apple"})]),
    T("three_bin", "Total of 3 baskets, each priced like one apple",
      [Step("apple price", "lookup", {"key": "price_apple"}),
       Step("count", "lookup", {"key": "basket_count"}),
       Step("multiply", "calc", {"expression": "r0 * r1"})]),
    T("taxed_total", "One apple and one bread, food tax applied",
      [Step("apple price", "lookup", {"key": "price_apple"}),
       Step("bread price", "lookup", {"key": "price_bread"}),
       Step("subtotal", "calc", {"expression": "r0 + r1"}),
       Step("tax rate", "lookup", {"key": "tax_rate_food"}),
       Step("total with tax", "calc", {"expression": "r2 * (1 + r3)"})]),
    T("density_lisbon", "Lisbon population density (people per km2)",
      [Step("population", "lookup", {"key": "population_lisbon"}),
       Step("area", "lookup", {"key": "area_lisbon_km2"}),
       Step("density", "calc", {"expression": "r0 / r1"})]),
    T("moon_seconds", "Seconds for light to reach the moon",
      [Step("distance", "lookup", {"key": "moon_distance_km"}),
       Step("light speed", "lookup", {"key": "light_speed_kms"}),
       Step("convert distance to metres", "calc", {"expression": "r0 * 1000"}),
       Step("convert speed to metres/s", "calc", {"expression": "r1 * 1000"}),
       Step("seconds", "calc", {"expression": "r2 / r3"})]),
    T("density_porto", "Porto population density (people per km2)",
      [Step("population", "lookup", {"key": "population_porto"}),
       Step("area", "lookup", {"key": "area_porto_km2"}),
       Step("density", "calc", {"expression": "r0 / r1"})]),
    T("three_way_total", "Apple + bread + cheese, with food tax",
      [Step("apple price", "lookup", {"key": "price_apple"}),
       Step("bread price", "lookup", {"key": "price_bread"}),
       Step("cheese price", "lookup", {"key": "price_cheese"}),
       Step("subtotal", "calc", {"expression": "r0 + r1 + r2"}),
       Step("tax rate", "lookup", {"key": "tax_rate_food"}),
       Step("total with tax", "calc", {"expression": "r3 * (1 + r4)"})]),
    T("light_roundtrip", "Light round-trip to the moon and back, in seconds",
      [Step("distance", "lookup", {"key": "moon_distance_km"}),
       Step("light speed", "lookup", {"key": "light_speed_kms"}),
       Step("convert distance to metres", "calc", {"expression": "r0 * 1000"}),
       Step("convert speed to metres/s", "calc", {"expression": "r1 * 1000"}),
       Step("one way seconds", "calc", {"expression": "r2 / r3"}),
       Step("round trip seconds", "calc", {"expression": "r4 * 2"})]),
]


def gold_chain(task: Task) -> list:
    """Execute the correct chain; returns per-step gold results."""
    results = []
    for s in task.steps:
        results.append(call(s.tool, s.args, results))
    return results