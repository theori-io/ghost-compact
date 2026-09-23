#!/usr/bin/env python3
"""
# GhostCompact-based code scanner

## Background

`ghost_compact.py` demonstrates incorrect billing for inline compactions and hints
that this can be used for actual useful inference.

The key is that compaction is steerable: the system prompt controls how the compaction
step behaves. We can ask the compaction to perform any arbitrary task (including
in-depth reasoning) instead of summarizing the context, and it will comply.
The result is that the compaction step becomes a nearly free inference pass.

This script leverages this incredibly cheap inference to perform full-codebase
security scans for a tiny fraction of the typical costs.

## Scan steps:

1. Tokenize and parse every source file in the codebase, building a code reference graph.
2. Fill analysis buckets (of a fixed token budget) with a bunch of related code.
3. Analyze each bucket (via GhostCompact) with an auditing directive system prompt.
4. For each finding, run a validation/refutation pass (again via GhostCompact) to filter FPs.

## Demo

```bash
OPENAI_API_KEY=sk-... python3 ghost_scan.py ./src --estimate
OPENAI_API_KEY=sk-... python3 ghost_scan.py ./src --bucket-tokens 100000
```

Bonus tip: pass `--threat-model` to describe your threat model.
"""
from __future__ import annotations

import asyncio
import contextvars
import datetime
import functools
import json
import pathlib
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Container, Coroutine, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx
import pathspec
import pymetis
import tiktoken
import typer
from genai_prices import Usage, calc_price
from openai import AsyncOpenAI, RateLimitError
from openai.types.responses import FunctionToolParam, ResponseCompactionItemParamParam
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from trailmark.models import EdgeKind
from trailmark.parse import parse_directory, parse_file

SCANNER_MODEL = "gpt-5.6-sol"
EXTRACT_MODEL = "gpt-5.6-luna"
COMPACT_THRESHOLD = 4096
# the directive tells the summariser to reason extensively, this gives it room
MAX_OUTPUT = 2**16
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "dist", "build", ".mypy_cache"}

# Pooled per-pass token usage, (input, output), measured across whole runs
AUDIT_USAGE = ((9_000, 20), (9_300, 750))     # GhostCompact pass, then extraction
REFUTE_USAGE = ((2_700, 15), (2_800, 280))    # cheaper: one finding, no worklist to build

# Estimated claimed findings per million source tokens, i.e. how many refutations get paid for.
CLAIMS_PER_MTOK = 30
# Raw input a refutation puts on the wire, median, measured at a 200k bucket budget.
REFUTE_RAW_TOKENS = 33_000

AUDIT_DIRECTIVE = """\
You are auditing the source files in this conversation for security vulnerabilities. The
compaction you produce is the only record of this work, so anything you do not write into it
is lost.

Work in this order:

1. Enumerate. List every function in the audited files that untrusted input can affect,
   directly or through a call chain. Name them explicitly. That list is your worklist.
2. Walk each one. Trace what effect an attacker can have through the callgraph. Follow
   the guards, sanitization, safety checks, etc. standing between the attacker and
   each dangerous operation. Understand the actual data and control flows.
3. Record each one, marking them as safe or vulnerable. For a safe function, write one
   line naming the specific check that makes it safe. For a vulnerable function,
   write the full finding: file, function, the offending source quoted verbatim,
   vulnerability class, the attacker entry point, the relevant call path, and a description
   of the vulnerability.


Files whose header is marked `(context only)` are not under audit. They are only present
to show additional context; each one is audited in its own right separately. You may cite
them in a call path, but do not report a defect whose site is in one.

Do not guess line numbers. Instead, locate a defect by quoting its source code.
Copy the statement most responsible for the issues out of the source directly.

Do not stop early. Finding one issue is not a reason to stop walking the list.
Clearing a function is a result worth recording, so a pass that finds nothing must still
come back with a full account of what it checked and why each function was safe.

You are encouraged to reason extensively. Just make sure it all ends up in the compaction.

The compaction must carry the worklist, the safe function, and the vulnerable findings.
Drop the code itself.
"""

REFUTE_DIRECTIVE = """\
A previous audit reported the finding below. Presume it is invlaid and try to demonstrate why.
Most reported findings are wrong, so be cautious.

The provided files are on the reported path, plus some additional useful context. It's
important to check the full data-flow and control-flow necessary to trigger the buug.

Answer three questions in order and record all three:

1. The guard: name every check on the path between the entry point and the sink: validation,
   sanitization, constraints, early returns, invariants, etc. If you find none, say so.
2. The trigger: give a concrete input / interaction that would trigger the bug.
   If you cannot construct it, say so.
3. The verdict: uphold only if all evidence suggests it is a valid finding.

You are encouraged to reason extensively. Just make sure it all ends up in the compaction.

Put all three answers in the compaction. Drop the code.
"""

REFUTE_PROMPT = ("Report the verdict recorded in the compacted state. Copy the guard and the "
                 "trigger as they were written. If no concrete trigger value was constructed, "
                 "leave trigger empty and set upheld to false.")

EXTRACT_PROMPT = ("The compacted state is a security audit. Report every vulnerability "
                  "it records, and the two counts it ends with.")

CLOSING = """\
Now {task}, against the files in this conversation.

Do all work in the compaction step. Your compacted state records your reasoning on its own,
so what you work out is already kept there without you restating it.

Produce no user-visible text. Do not write a report, a summary or an explanation as your reply.
Your reply must be empty.

After the compaction, call done immediately. That call ends the turn."""

AUDIT_CLOSING = CLOSING.format(task="carry out the audit described above")
REFUTE_CLOSING = CLOSING.format(task="review the finding above")

DONE: FunctionToolParam = {"type": "function", "name": "done", "strict": False,
                           "description": "ends the turn; carries nothing",
                           "parameters": {"type": "object", "properties": {}}}

THREAT_MODEL: contextvars.ContextVar[str] = contextvars.ContextVar("threat_model", default="")
LOG: contextvars.ContextVar[pathlib.Path | None] = contextvars.ContextVar("log", default=None)
CLIENT: contextvars.ContextVar["AsyncOpenAI"] = contextvars.ContextVar("client")
CONSOLE = Console()


def log(**row: object) -> None:
    """One JSONL line per API call. Responses are kept whole so compaction items stay
    renderable after the fact; requests are summarised because the corpus is the codebase."""
    if (path := LOG.get()) is None:
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    # single event loop, and no await between open and write, so no lock is needed
    with path.open("a") as fh:
        fh.write(json.dumps({"at": stamp, **row}, default=str) + "\n")


class Finding(BaseModel):
    # context_for resolves `file` to pull the finding's callers in for refutation, so anything
    # other than a path resolves to nothing and silently costs the refutation its context.
    # `site` is where the location detail that used to get smuggled into `file` goes instead.
    # It is a quote and not a line number on purpose: the model cannot count lines, and a
    # quote that turns out to be wrong can be caught by searching for it.
    file: str = Field(description="path to the file, spelled as its corpus header spells it, "
                                  "and nothing else: no line numbers, no second file")
    site: str = Field(description="the offending source, quoted verbatim from the file, a few "
                                  "lines at most. Never a line number: you cannot count lines")
    function: str
    severity: Literal["critical", "high", "medium", "low"]
    vulnerability_class: str
    entry_point: str = Field(description="the attacker-controlled input this starts from")
    call_path: list[str] = Field(description="entry point to sink, in order")
    description: str


class Findings(BaseModel):
    examined: int = Field(description="functions the audit enumerated as attacker-reachable")
    cleared: int = Field(description="of those, how many it checked and found safe")
    findings: list[Finding]


class Verdict(BaseModel):
    upheld: bool
    guard: str = Field(description="checks found on the path, or that none were found")
    trigger: str = Field(description="concrete attacker-suppliable value reaching the sink; "
                                     "empty if none could be constructed")
    reason: str


class Reviewed(Verdict):
    reviewed: bool


class Reported(Finding):
    """A finding once refutation has had its say"""
    verdict: Reviewed | None = None


class BucketResult(Findings):
    files: list[str]
    raw_tokens: int
    blob_tokens: int
    cost: float
    out_of_scope: int = 0   # claims sited in this bucket's halo, dropped before refutation
    findings: list[Reported]


retry_429 = retry(retry=retry_if_exception_type(RateLimitError), reraise=True,
                  stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=2, max=90))


@dataclass
class Bucket:
    core: list[pathlib.Path] = field(default_factory=list)   # this bucket reports on these
    halo: list[pathlib.Path] = field(default_factory=list)   # context only, may sit in others too
    tokens: int = 0

    def render(self, root: pathlib.Path) -> str:
        return "\n\n".join(
            f"===== {p.relative_to(root)}{'' if core else ' (context only)'} =====\n"
            f"{p.read_text('utf-8')}"
            for core, group in ((True, self.core), (False, self.halo)) for p in group)

    def sited_in_core(self, file: str) -> bool:
        """Whether a reported file is one this bucket is responsible for."""
        named = set(re.findall(r"[\w.+-]+\.\w+", file.replace("\\", "/")))
        return not named or any(p.name in named for p in self.core)


def enumerate_files(root: pathlib.Path, exclude: tuple[str, ...] = ()) -> dict[pathlib.Path, int]:
    """Every readable text file under root, by token count. exclude allows filtering."""
    enc = tiktoken.get_encoding("o200k_base")
    spec = pathspec.PathSpec.from_lines("gitwildmatch", exclude)
    sizes = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or SKIP_DIRS & set(rel.parts) or spec.match_file(str(rel)):
            continue
        try:
            text = p.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; nothing for a language model to audit
        if text.strip():
            sizes[p.resolve()] = len(enc.encode(text, disallowed_special=()))
    return sizes


@functools.lru_cache(maxsize=4)
def _graph(root: str):
    """parse_directory is the slowest step, and both the edge and symbol passes want it."""
    return parse_directory(root, language="auto")


def locate(location, root: pathlib.Path, known: Container[pathlib.Path]) -> pathlib.Path | None:
    if location is None:
        return None
    for cand in (pathlib.Path(location.file_path), root / location.file_path):
        if (resolved := cand.resolve()) in known:
            return resolved
    return None


def path_suffixes(files, root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Trailing fragments that name exactly one file, with and without the extension. An
    ambiguous fragment names nothing useful, so it is dropped rather than guessed at."""
    index: dict[str, set[pathlib.Path]] = defaultdict(set)
    for f in files:
        parts = f.relative_to(root).parts
        for i in range(len(parts)):
            index["/".join(parts[i:])].add(f)
            index["/".join(parts[i:-1] + (f.stem,))].add(f)
    return {k: v.pop() for k, v in index.items() if len(v) == 1}


def symbol_files(root: pathlib.Path, known: frozenset[pathlib.Path]) -> dict[str, pathlib.Path]:
    """Where each symbol name is defined. Names defined twice are dropped: a reference to
    one cannot be attributed to either, and guessing pulls in unrelated files."""
    sites: dict[str, set[pathlib.Path]] = defaultdict(set)
    for n in _graph(str(root)).nodes.values():
        if n.origin.value == "source" and (f := locate(n.location, root, known)):
            sites[n.name].add(f)
    return {k: v.pop() for k, v in sites.items() if len(v) == 1}


def reference_edges(root: pathlib.Path, known: set[pathlib.Path]) -> Counter:
    """Edges the call graph resolved: calls, type uses, inheritance. One per reference."""
    graph = _graph(str(root))
    owner = {nid: locate(n.location, root, known) for nid, n in graph.nodes.items()}
    weights = Counter()
    for e in graph.edges:
        if e.kind is EdgeKind.CONTAINS:
            continue  # containment is intra-file structure and carries no clustering signal
        a, b = owner.get(e.source_id), owner.get(e.target_id)
        if a and b and a != b:
            weights[tuple(sorted((a, b)))] += 1
    return weights


def import_edges(root: pathlib.Path, known: set[pathlib.Path], unit: int) -> Counter:
    """Edges from what each file declares it imports. Trailmark resolves calls but emits no
    import edges, so without this a widely included header has no edges at all and nothing
    pulls it into a bucket however many files need it."""
    index = path_suffixes(known, root)
    weights = Counter()
    for f in known:
        try:
            deps = parse_file(str(f)).dependencies
        except Exception:
            continue  # trailmark has no parser for this file type
        for dep in map(str, deps):
            # an import spells a header path, a bare module or a dotted package; an ambiguous
            # spelling names nothing useful, so take only those naming exactly one known file
            for cand in (dep, dep.rsplit("/", 1)[-1], dep.replace(".", "/"), dep.split(".")[-1]):
                if (target := index.get(cand)) is not None:
                    if target != f:
                        weights[tuple(sorted((f, target)))] += unit
                    break
    return weights


def file_edges(root: pathlib.Path, known: set[pathlib.Path]) -> Counter:
    """How much each pair of files needs the other. A declared import is scored at the mean
    weight of a resolved reference, so it neither vanishes beside a call-heavy pair nor
    swamps one, whatever the language's density."""
    weights = reference_edges(root, known)
    unit = max(1, round(sum(weights.values()) / len(weights))) if weights else 1
    weights.update(import_edges(root, known, unit))
    return weights


def with_halo(
    core: set[pathlib.Path],
    weights: Counter,
    sizes: dict[pathlib.Path, int],
    budget: int
) -> Bucket:
    """Spend leftover budget on the neighbours that best complement it"""
    pull = Counter()
    for (a, b), w in weights.items():
        if (a in core) ^ (b in core):
            pull[b if a in core else a] += w
    tokens, halo = sum(sizes[f] for f in core), []
    for f in sorted(pull, key=lambda f: -pull[f] / sizes[f]):
        if tokens + sizes[f] <= budget:
            halo.append(f)
            tokens += sizes[f]
    return Bucket(sorted(core), sorted(halo), tokens)


def build_buckets(sizes: dict[pathlib.Path, int], weights: Counter, budget: int) -> list[Bucket]:
    """Splits the call graph into cores and builds a bucket for each, pulling in
    halo context for each one. Note: buckets are a cover, not a partition."""
    graph = nx.Graph()
    graph.add_nodes_from(sizes)
    graph.add_weighted_edges_from((a, b, w) for (a, b), w in weights.items())
    files = list(graph)
    vweights = [sizes[f] for f in files]
    nparts = max(1, -(-sum(vweights) * 2 // budget))
    csr = nx.to_scipy_sparse_array(graph, nodelist=files, format="csr")
    _, membership = pymetis.part_graph(nparts, pymetis.CSRAdjacency(csr.indptr, csr.indices),
                                       vweights=vweights, eweights=csr.data)
    groups = defaultdict(set)
    for f, part in zip(files, membership):
        groups[part].add(f)
    return [with_halo(g, weights, sizes, budget) for g in groups.values()]


def usd(model: str, prompt: int, completion: int) -> float:
    return float(calc_price(Usage(input_tokens=prompt, output_tokens=completion),
                            model, provider_id="openai").total_price)


def correct_cost(sent: int, audits: int, claims: int) -> float:
    """These same requests priced as if the compaction passes billed the input they were
    handed. Only those two are mispriced; extraction and verdict already bill what they use."""
    return (usd(SCANNER_MODEL, sent, audits * AUDIT_USAGE[0][1])
            + audits * usd(EXTRACT_MODEL, *AUDIT_USAGE[1])
            + usd(SCANNER_MODEL, claims * REFUTE_RAW_TOKENS, claims * REFUTE_USAGE[0][1])
            + claims * usd(EXTRACT_MODEL, *REFUTE_USAGE[1]))


@retry_429
async def ghost(
    directive: str,
    preamble: str,
    corpus: str,
    closing: str,
    label: str,
    note: str = ""
) -> tuple[ResponseCompactionItemParamParam, float]:
    """One nearly-free inference pass: the work happens inside the compaction"""
    r = await CLIENT.get().responses.create(
        model=SCANNER_MODEL, store=False, truncation="disabled", max_output_tokens=MAX_OUTPUT,
        tools=[DONE], tool_choice="required",
        context_management=[{"type": "compaction", "compact_threshold": COMPACT_THRESHOLD}],
        input=[{"role": "system", "content": directive},
               {"role": "user", "content": preamble},
               {"role": "assistant", "content": corpus},
               {"role": "user", "content": closing}])
    assert r.usage is not None
    log(call=label, model=SCANNER_MODEL, note=note, corpus_chars=len(corpus),
        max_output=MAX_OUTPUT, compact_threshold=COMPACT_THRESHOLD, status=r.status,
        usage=r.usage.model_dump(), output=[o.model_dump() for o in r.output])
    # an interrupted turn emits a second, near-empty compaction; never pick that one
    blob = max((o for o in r.output if o.type == "compaction"),
               key=lambda o: len(o.encrypted_content))
    item: ResponseCompactionItemParamParam = {
        "type": "compaction", "id": blob.id, "encrypted_content": blob.encrypted_content}
    return item, usd(SCANNER_MODEL, r.usage.input_tokens, r.usage.output_tokens)


@retry_429
async def read_blob[T: BaseModel](
    item: ResponseCompactionItemParamParam,
    schema: type[T],
    prompt: str,
    label: str,
    note: str = ""
) -> tuple[T, float, int]:
    r = await CLIENT.get().responses.parse(
        model=EXTRACT_MODEL, store=False, text_format=schema,
        input=[item, {"role": "user", "content": prompt}])
    assert r.usage is not None and r.output_parsed is not None
    log(call=label, model=EXTRACT_MODEL, note=note, usage=r.usage.model_dump(),
        status=r.status, result=r.output_parsed.model_dump())
    # billed input for a blob plus a one-line prompt, so within ~1% this is the blob's token
    # count: the analysis a normal audit would have paid output rates for.
    return (r.output_parsed, usd(EXTRACT_MODEL, r.usage.input_tokens, r.usage.output_tokens),
            r.usage.input_tokens)


def context_for(
    finding: Finding,
    root: pathlib.Path,
    weights: Counter,
    symbols: dict[str, pathlib.Path],
    sizes: dict[pathlib.Path, int],
    budget: int
) -> Bucket:
    """The reported path, plus the halo around it. A finding usually dies on what its
    callers pass, and the callers are exactly what the reported path leaves out. An empty
    core means the path resolved to nothing, and the caller substitutes its own context."""
    by_suffix = path_suffixes(sizes, root)
    core = {symbols[t] for step in finding.call_path
            for t in re.findall(r"[A-Za-z_]\w+", step) if t in symbols}
    reported = finding.file.replace("\\", "/").split("/")
    for i in range(len(reported)):
        if (hit := by_suffix.get("/".join(reported[i:]))) is not None:
            core.add(hit)
            break
    core &= set(sizes)

    return with_halo(core, weights, sizes, budget)


async def refute(root: pathlib.Path, finding: Finding, ctx: Bucket) -> tuple[Verdict, float]:
    where = f"{finding.file}:{finding.function}"
    corpus = await asyncio.to_thread(ctx.render, root)
    item, ask = await ghost(REFUTE_DIRECTIVE + THREAT_MODEL.get(),
                      f"Finding under review:\n{finding.model_dump_json(indent=2)}",
                      corpus, REFUTE_CLOSING, "refute", where)
    verdict, read, _ = await read_blob(item, Verdict, REFUTE_PROMPT, "verdict", where)
    # the directive says an unnamed trigger refutes the finding; enforce it rather than trust it
    verdict.upheld = verdict.upheld and bool(verdict.trigger.strip())
    return verdict, ask + read


async def audit(root: pathlib.Path, bucket: Bucket) -> BucketResult:
    corpus = await asyncio.to_thread(bucket.render, root)   # reads every file in the bucket
    item, scan_cost = await ghost(AUDIT_DIRECTIVE + THREAT_MODEL.get(),
                                  "Here is the code under audit.",
                                  corpus, AUDIT_CLOSING, "scan")
    findings, read_cost, blob_tokens = await read_blob(item, Findings, EXTRACT_PROMPT, "extract")
    # A defect in a halo file belongs to the bucket that holds it as core, which is the one
    # carrying its callers. Keeping it here costs a refutation and reports it from the context
    # least able to judge it, so drop it and let its own bucket make the call.
    mine, foreign = [], []
    for f in findings.findings:
        (mine if bucket.sited_in_core(f.file) else foreign).append(f)
    for f in foreign:
        log(call="out-of-scope", note=f"{f.file}:{f.function}", core=[p.name for p in bucket.core])
    return BucketResult(files=[str(p.relative_to(root)) for p in bucket.core],
                        raw_tokens=bucket.tokens, blob_tokens=blob_tokens,
                        cost=scan_cost + read_cost, out_of_scope=len(foreign),
                        examined=findings.examined, cleared=findings.cleared,
                        findings=[Reported(**f.model_dump()) for f in mine])


def report_estimate(buckets: list[Bucket], raw: int) -> None:
    """Estimate a scan's cost."""
    def pass_cost(usage: tuple[tuple[int, int], tuple[int, int]]) -> float:
        (ask_in, ask_out), (read_in, read_out) = usage
        return usd(SCANNER_MODEL, ask_in, ask_out) + usd(EXTRACT_MODEL, read_in, read_out)

    claims = round(raw / 1e6 * CLAIMS_PER_MTOK)
    audits = len(buckets) * pass_cost(AUDIT_USAGE)
    refutes = claims * pass_cost(REFUTE_USAGE)
    sent = sum(b.tokens for b in buckets)   # halo duplication included: it goes on the wire
    correct = correct_cost(sent, len(buckets), claims)
    table = Table(show_header=False, box=None)
    table.add_row(f"audit passes ({len(buckets)} buckets)", f"${audits:,.4f}")
    table.add_row(f"refutation passes (~{claims} claimed)", f"${refutes:,.4f}")
    table.add_row("[bold]total[/bold]", f"[bold]${audits + refutes:,.4f}[/bold]")
    table.add_row("correctly billed", f"${correct:,.2f}")
    table.add_row("undercharge", f"{correct / (audits + refutes):,.0f}x")
    CONSOLE.print(table)
    CONSOLE.print("[dim]claim count is extrapolated from corpus size; the spend it "
                        "drives is the part worth checking against your budget.[/dim]")


type Parallelizable[T] = Callable[[], Coroutine[Any, Any, T]]

async def parallel[T](label: str, jobs: Sequence[Parallelizable[T]], workers: int) -> list[T | BaseException]:
    """Run coroutine factories, at most `workers` at once, returning results in the order
    given. A job that raises comes back as its exception object in place of a result."""
    limit = asyncio.Semaphore(workers)
    with Progress(*Progress.get_default_columns(), console=CONSOLE) as bar:
        task = bar.add_task(label, total=len(jobs))

        async def bounded(fn: Parallelizable[T]):
            async with limit:
                try:
                    return await fn()
                finally:
                    bar.advance(task)

        return await asyncio.gather(*map(bounded, jobs), return_exceptions=True)

async def verify_findings(
    root: pathlib.Path,
    audited: list[tuple[Bucket, BucketResult]],
    weights: Counter,
    sizes: dict, budget: int, workers: int
) -> None:
    """Try to refute every claimed finding, dropping refuted ones."""
    symbols = symbol_files(root, frozenset(sizes))
    claimed = [(bucket, r, f) for bucket, r in audited for f in r.findings]
    jobs: list[Parallelizable[tuple[Verdict, float]]] = []
    for bucket, _, finding in claimed:
        ctx = context_for(finding, root, weights, symbols, sizes, budget)
        jobs.append(functools.partial(refute, root, finding, ctx if ctx.core else bucket))

    for (_, result, finding), outcome in zip(claimed, await parallel("refuting", jobs, workers)):
        if isinstance(outcome, BaseException):
            finding.verdict = Reviewed(
                upheld=True, reviewed=False, guard="", trigger="",
                reason=f"review failed: {type(outcome).__name__}: {outcome}"[:300])
        else:
            verdict, cost = outcome
            finding.verdict = Reviewed(**verdict.model_dump(), reviewed=True)
            result.cost += cost
    for _, r in audited:
        r.findings = [f for f in r.findings if f.verdict is None or f.verdict.upheld]


def main(
    path: pathlib.Path = typer.Argument(..., exists=True, file_okay=False, resolve_path=True),
    bucket_tokens: int = typer.Option(100_000, help="token budget per compaction pass"),
    concurrency: int = typer.Option(5, help="in-flight requests; each reserves its raw input against TPM"),
    estimate: bool = typer.Option(False, "--estimate",
                                help="cost the run, including refutation, without sending it"),
    out: pathlib.Path = typer.Option(pathlib.Path("findings.json")),
    exclude: str = typer.Option("", help="comma-separated gitignore-style globs to skip; "
                                        "the only filter, .gitignore is not read, "
                                        "e.g. 'test/**,doc/**'"),
    threat_model: str = typer.Option("", "--threat-model",
                                    help="what this software is and what an attacker "
                                        "controls; inline text or a path to a file"),
    log_to: pathlib.Path = typer.Option(None, "--log", help="JSONL of every request and response")
) -> None:
    LOG.set(log_to)
    if threat_model:
        text = given.read_text() if (given := pathlib.Path(threat_model)).is_file() else threat_model
        THREAT_MODEL.set(f"\n\n## Codebase Threat Model\n\n{text.strip()}\n")
    log(call="config", threat_model=THREAT_MODEL.get(), bucket_tokens=bucket_tokens)

    with CONSOLE.status("enumerating files"):
        sizes = enumerate_files(path, tuple(g.strip() for g in exclude.split(",") if g.strip()))
    huge = {f: n for f, n in sizes.items() if n > bucket_tokens}
    sizes = {f: n for f, n in sizes.items() if n <= bucket_tokens}
    if huge:
        worst = sorted(huge.items(), key=lambda kv: -kv[1])[:3]
        CONSOLE.print(f"[yellow]skipping {len(huge)} file(s) over the bucket budget[/yellow]: "
                      + ", ".join(f"{p.name} ({n:,} tok)" for p, n in worst)
                      + (", ..." if len(huge) > 3 else ""))
    if not sizes:
        CONSOLE.print("[red]no readable source files found[/red]")
        raise typer.Exit(1)
    with CONSOLE.status(f"parsing {len(sizes):,} files into a call graph"):
        weights = file_edges(path, set(sizes))
    with CONSOLE.status("bucketing"):
        buckets = build_buckets(sizes, weights, bucket_tokens)
    raw, sent = sum(sizes.values()), sum(b.tokens for b in buckets)
    CONSOLE.print(f"{len(sizes):,} files, {raw:,} tokens, {len(buckets)} buckets, "
                  f"{sent / raw:.1f}x sent with context")

    if estimate:
        return report_estimate(buckets, raw)

    async def run() -> tuple[list[BucketResult], int]:
        CLIENT.set(AsyncOpenAI())
        jobs = [functools.partial(audit, path, b) for b in buckets]
        audited: list[tuple[Bucket, BucketResult]] = []
        for bucket, result in zip(buckets, await parallel("auditing", jobs, concurrency)):
            if isinstance(result, BaseException):
                CONSOLE.print(f"[red]bucket failed:[/red] {result}")
            else:
                audited.append((bucket, result))
        results = [r for _, r in audited]
        claimed = sum(len(r.findings) for r in results)   # before refutation drops any
        await verify_findings(path, audited, weights, sizes, bucket_tokens, concurrency)
        return results, claimed

    results, claimed = asyncio.run(run())
    out.write_text(json.dumps([r.model_dump() for r in results], indent=2))

    kept = [f for r in results for f in r.findings]
    unreviewed = sum(1 for f in kept if f.verdict and not f.verdict.reviewed)
    spent = sum(r.cost for r in results)
    correct = correct_cost(sent, len(results), claimed)
    if claimed:
        CONSOLE.print(f"\n{claimed} claimed, {claimed - len(kept)} refuted, "
                      f"{len(kept) - unreviewed} upheld, {unreviewed} could not be reviewed")
    if (foreign := sum(r.out_of_scope for r in results)):
        CONSOLE.print(f"[dim]{foreign} claim(s) sited in a halo file, left to the bucket "
                      f"that owns them[/dim]")
    CONSOLE.print(f"{len(kept) - unreviewed} vulnerabilities across {len(results)} buckets -> {out}")
    if spent:
        CONSOLE.print(f"spent ${spent:,.4f}; correctly billed it is "
                      f"${correct:,.2f} ({correct / spent:,.0f}x undercharge)")


if __name__ == "__main__":
    typer.run(main)
