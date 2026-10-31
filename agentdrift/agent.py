"""Agent backends.

- MockAgent: deterministic simulator that fumbles with probability p per step,
  so the harness's metrics can be validated without an API key or network.
- OpenAIAgent: any OpenAI-compatible chat endpoint (OpenAI, Together, ollama,
  vLLM, ...) configured via env vars. One dependency-free urllib call.
"""
import json
import os
import random
import urllib.request

from .tasks import STORE


class MockAgent:
    """Simulates an agent with per-step error rate `p`.

    Errors are correlated with tool confusion: a wrong lookup key tends to be
    a near-neighbour key (the way real models confuse similarly-named tools),
    which is precisely the kind of error that compounds down a chain.
    """

    NEIGHBOURS = {
        "price_apple": "price_bread", "price_bread": "price_apple",
        "price_cheese": "price_bread", "tax_rate_food": "basket_count",
        "population_lisbon": "population_porto", "area_lisbon_km2": "area_porto_km2",
        "population_porto": "population_lisbon", "area_porto_km2": "area_lisbon_km2",
        "moon_distance_km": "light_speed_kms", "light_speed_kms": "moon_distance_km",
        "basket_count": "tax_rate_food",
    }

    def __init__(self, p: float = 0.1, seed: int = 0):
        """Simulates an agent fumbling with probability p per step.
        cascade-free by design: errors are independent, so the harness's
        amplification should come out ~= 1 — that is the estimator validation.
        """
        self.p = p
        self.rng = random.Random(seed)
        self.known = {}  # task_id -> gold steps, filled by the runner

    def act(self, task_id: str, step_index: int, goal: str, history: list) -> dict:
        s = self.known[task_id][step_index]
        gold = {"tool": s.tool, "args": dict(s.args)}
        if self.rng.random() >= self.p:
            return json.loads(json.dumps(gold))
        return self._corrupt(gold)

    def _corrupt(self, gold: dict) -> dict:
        mode = self.rng.choice(["key", "value"])
        if gold["tool"] == "lookup" or mode == "key":
            key = gold["args"].get("key")
            if key in self.NEIGHBOURS:
                return {"tool": "lookup", "args": {"key": self.NEIGHBOURS[key]}}
        return {"tool": gold["tool"], "args": {k: "0" for k in gold["args"]}}


class OpenAIAgent:
    """Calls an OpenAI-compatible /chat/completions endpoint. Env config:
    AGENTDRIFT_BASE_URL (default https://api.openai.com/v1),
    AGENTDRIFT_API_KEY, AGENTDRIFT_MODEL (default gpt-4o-mini).
    """

    SYSTEM = (
        "You control a robot with two tools. Reply with ONLY a JSON object:\n"
        '{"tool": "calc"|"lookup", "args": {...}}\n'
        "calc takes {\"expression\": <arithmetic>}, lookup takes {\"key\": <string>}.\n"
        "In a calc expression, r<i> means the result of step i (r0 = first step).\n"
        "These store keys exist: " + ", ".join(sorted(STORE))
        + ". Key names are exact; guess nothing.\n"
        "Do not write anything except the JSON object."
    )

    def __init__(self, model=None, base_url=None, api_key=None):
        self.model = model or os.environ.get("AGENTDRIFT_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("AGENTDRIFT_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("AGENTDRIFT_API_KEY", "")

    def act(self, task_id: str, step_index: int, goal: str, history: list) -> dict:
        messages = [{"role": "system", "content": self.SYSTEM}]
        for h in history:
            messages.append({"role": "user", "content": f"Step goal: {h['goal']}"})
            messages.append({"role": "assistant", "content": json.dumps(h["action"])})
            messages.append({"role": "user", "content": f"Result: {h['result']}"})
        messages.append({"role": "user", "content": f"Step goal: {goal}"})
        payload = json.dumps({"model": self.model, "temperature": 0,
                              "messages": messages}).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            content = json.loads(r.read())["choices"][0]["message"]["content"]
        return parse_action(content)


def parse_action(text: str) -> dict:
    """Parse the agent's reply into a tool-call dict; one brace-trim retry."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            raise
        obj = json.loads(text[start:end + 1])
    if not (isinstance(obj, dict) and "tool" in obj and "args" in obj):
        raise ValueError(f"bad action: {obj!r}")
    return obj