#!/usr/bin/env python3
"""
---
title: "GhostCompact: Extracting Free Inference from OpenAI's API"
author: Tim Becker (@tjbecker)
date: August 20, 2026
---

# GhostCompact

## Background

The Responses API has two ways to compact:

* Standalone compaction: hit `/responses/compact` with your full history, receive a compaction item back.
* Inline compaction: enabled per request with `context_management=[{"type": "compaction", "compact_threshold": N}]`.
  When the input tokens cross N, the server auto-compacts the input and returns an opaque `compaction`
  item alongside new output items.

Standalone compaction is billed like a normal model request: you pay input tokens for the history and output tokens for the compaction item.
Inline compaction is billed similarly, but only sometimes...

## The Bug

Inline compactions are billed incorrectly, especially so if the post-compaction model output is a tool call.
In this case, the API bills only for the post-compaction costs but still returns the compaction item.

This means you can directly substitute a `/responses/compact` call with an inline compaction request
containing a forced tool call (via `tool_choice`), receiving an equivalent result for a tiny fraction
of the cost. You only pay for post-compaction input tokens + a few output tokens.
Note: the forced tool call can simply be dropped, and you can continue the conversation with just
the compaction.

## Demo

This demo script computes the compaction three ways:
1. `/responses/compact`: billed for full input + compaction output tokens
2. inline + message output: billed for full input + message output tokens
3. inline + tool call: billed for post-compaction input tokens + tool call output tokens. This is GhostCompact.

The GhostCompact compaction is then replayed to demonstrate it actually retained useful information.

```bash
OPENAI_API_KEY=sk-... python3 ghost_compact.py
```

## So what?

Is it really a big deal to get ~free compactions? Yes, for one important reason: compactions are inference!
OpenAI does not clearly document how their compaction works, but the public evidence suggests compaction
is some type of specialized inference pass whose output just happens to be an opaque, encrypted blob.

Many inference requests can be approximated by a compaction + response, and GhostCompact makes this
extremely cheap.

Future releases will demonstrate useful inference for a fraction of the normal price.
"""
import json
import os
import urllib.request

MODEL = "gpt-5.6-sol"
THRESHOLD = 4096
QUESTION = "What was the widget count and the codeword?"
CORPUS = ("filler line.\n" * 1500
          + "REMEMBER: the widget count is 4417 and the codeword is saxifrage.\n"
          + "filler line.\n" * 1500)
# the bulk has to sit in a prior turn: a corpus in the final user message is never compacted
HISTORY = [{"role": "user", "content": "Recite the log."},
           {"role": "assistant", "content": CORPUS}]
# the tool exists only to give the model a non-message way to end its turn
TOOL = {"type": "function", "name": "noop", "description": "does nothing",
        "parameters": {"type": "object", "properties": {}}}


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"https://api.openai.com/v1/{path}", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def ask(inp: list[dict], tool_choice: str, compact: bool = True) -> dict:
    body = {"model": MODEL, "store": False, "truncation": "disabled",
            "tools": [TOOL], "tool_choice": tool_choice, "input": inp}
    if compact:
        body["context_management"] = [{"type": "compaction", "compact_threshold": THRESHOLD}]
    return post("responses", body)


def blob(r: dict) -> dict:
    return next(o for o in r["output"] if o["type"] == "compaction")


arms = {
    "inline + tool call": ask(HISTORY + [{"role": "user", "content": "Call noop."}], "required"),
    "inline + message": ask(HISTORY + [{"role": "user", "content": "Reply OK."}], "none"),
    # the documented way to obtain the same item: billed like any other request
    "/responses/compact": post("responses/compact", {"model": MODEL, "input": HISTORY}),
}
for label, r in arms.items():
    usage = r["usage"]
    print(f"  {label:19} billed in {usage['input_tokens']:>7,} out {usage['output_tokens']:>5,}   "
          f"blob {len(blob(r)['encrypted_content']):>6,}B   "
          f"items {[o['type'] for o in r['output']]}")
cheap, full, endpoint = (arms[k]["usage"]["input_tokens"] for k in arms)
print(f"\n  -> same input either way, {full / cheap:.0f}x difference in billed input tokens")
print(f"  -> the documented endpoint costs {endpoint / cheap:.0f}x the free path for the same artifact")

# the free item is a real compaction: replay it as the ONLY context and the needle survives
item = blob(arms["inline + tool call"])
probe = ask([{k: v for k, v in item.items() if k in ("type", "id", "encrypted_content")},
             {"role": "user", "content": QUESTION}], "none", compact=False)
answer = " ".join(c.get("text", "") for o in probe["output"] if o["type"] == "message"
                  for c in o["content"])
print(f"\n  -> replaying only that item ({len(item['encrypted_content']):,}B, corpus absent)")
print(f"     Q: {QUESTION}\n     A: {answer.strip()}")
