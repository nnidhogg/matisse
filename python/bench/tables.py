"""Writes docs/python.md's tables of measurements from the outputs run.sh and run_tokenizer_json.sh keep, every figure
read from a row.

    python -I tables.py RESULTS_DIR

Each table is printed in Markdown with its columns aligned, the medians of the rows' passes, as the outputs give them;
a directory of run_tokenizer_json.sh's outputs gives the tables beside Hugging Face tokenizers, any other run.sh's.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROW = re.compile(
    r"^(?:(matisse-py|tiktoken|hf)\s+)?(\w+)\s+(.+?)\s+(\S+)\s+([\d.]+) MiB,\s+(\d+) (?:tokens|pieces), (\d+) passes: "
    r"best\s+([\d.]+), median\s+([\d.]+), worst\s+([\d.]+) MiB/s"
)
"""A row of matisse_bench or of bench.py: the vendor, absent for matisse_bench, the tokenizer, the label, the input, its
MiB, the ids' count, the passes and the best, median and worst MiB/s."""

CPP_LATENCY = re.compile(r"^(\w+)\s+(latency(?: kept)?)\s+(\S+)\s+(\d+) calls of \d+ bytes on average: per call mean "
                         r"([\d.]+)")
"""A latency row of matisse_bench: the tokenizer, the label, the input, the calls and the mean microseconds."""

PY_LATENCY = re.compile(r"^(matisse-py|tiktoken|hf)\s+(\w+)\s+(.+? latency)\s+(\S+)\s+(\d+) calls: per call median "
                        r"pass mean ([\d.]+) us")
"""A latency row of bench.py: the vendor, the tokenizer, the label, the input, the calls and the mean microseconds."""

LOAD = re.compile(r"^(matisse-py|tiktoken)\s+(\w+)\s+(load(?: \w+)?)\s+(\d+) processes: median ([\d.]+) s")
"""A load row of bench.py: the vendor, the tokenizer, the label, the processes and the median seconds."""

IDS = re.compile(r"^ids (\S+) (\w+) (\S+) (\d+) ([0-9a-f]{64})$")
"""An ids line: the vendor, the tokenizer, the input, the ids' count and their SHA-256."""

DIFFERS = re.compile(r"^differs (\w+) (\S+): not timed$")
"""bench.py's line for a tokenizer and an input whose ids differ between the two packages."""

TOKENIZERS = ["gpt2", "cl100k", "o200k", "llama3"]
"""The tokenizers, in the order the tables list them."""

CORPORA = {
    "twitter.json": "`twitter.json`",
    "munch-python@15b2863c.txt": "`munch-python@15b2863c`",
    "munch-cpp@15b2863c.txt": "`munch-cpp@15b2863c`",
    "gutenberg.txt": "`gutenberg`",
    "wiki-zh.txt": "`wiki-zh`",
    "wiki-ar.txt": "`wiki-ar`",
    "wiki-ru.txt": "`wiki-ru`",
    "wiki-hi.txt": "`wiki-hi`",
    "wiki-sv.txt": "`wiki-sv`",
    "enwik8": "`enwik8`",
    "prompts.txt": "`prompts`",
    "short.txt": "`prompts`, 48 bytes at most",
}
"""The inputs of the tables beside Hugging Face tokenizers, by file, in their order, with the names the tables give."""


def read(directory: pathlib.Path) -> tuple[dict, dict, dict, dict, set]:
    """Returns every row of the outputs in a directory.

    Args:
        directory: The directory.

    Returns:
        The rows' MiB and median MiB/s and count, the latencies in microseconds, and the load medians in seconds, each
        by vendor, tokenizer, label and input; the counts and digests of the ids lines by vendor, tokenizer and input;
        and the tokenizers and inputs whose ids differ between the packages.
    """
    rows: dict = {}
    latencies: dict = {}
    loads: dict = {}
    ids: dict = {}
    differs: set = set()
    for path in sorted(directory.glob("*.txt")):
        for line in path.read_text().splitlines():
            if match := ROW.match(line):
                vendor, tokenizer, label, name, mib, count, _, _, median, _ = match.groups()
                rows[(vendor or "matisse", tokenizer, label.strip(), name)] = (float(mib), float(median), int(count))
            elif match := CPP_LATENCY.match(line):
                tokenizer, label, name, _, mean = match.groups()
                latencies[("matisse", tokenizer, label, name)] = float(mean)
            elif match := PY_LATENCY.match(line):
                vendor, tokenizer, label, name, _, mean = match.groups()
                latencies[(vendor, tokenizer, label.strip(), name)] = float(mean)
            elif match := LOAD.match(line):
                vendor, tokenizer, label, _, median = match.groups()
                loads[(vendor, tokenizer, label)] = float(median)
            elif match := IDS.match(line):
                vendor, tokenizer, name, count, digest = match.groups()
                ids.setdefault((vendor, tokenizer, name), set()).add((int(count), digest))
            elif match := DIFFERS.match(line):
                differs.add(match.groups())
    return rows, latencies, loads, ids, differs


def table(header: list[str], body: list[list[str]], right: set[int]) -> str:
    """Returns a Markdown table with its columns padded to equal width.

    Args:
        header: The header's cells.
        body: The rows' cells.
        right: The columns aligned right.

    Returns:
        The table.
    """
    widths = [max(len(row[column]) for row in [header, *body]) for column in range(len(header))]

    def line(cells: list[str]) -> str:
        padded = [cell.rjust(width) if column in right else cell.ljust(width)
                  for column, (cell, width) in enumerate(zip(cells, widths))]
        return "| " + " | ".join(padded) + " |"

    rule = ["-" * width + (":" if column in right else "-") for column, width in enumerate(widths)]
    rule = "|" + "|".join("-" + cell for cell in rule) + "|"
    return "\n".join([line(header), rule, *(line(row) for row in body)])


def rate(rows: dict, key: tuple) -> str:
    """Returns a row's median MiB/s, or a dash when no row holds it.

    Args:
        rows: The rows.
        key: The row's key.

    Returns:
        The figure.
    """
    return f"{rows[key][1]:.1f}" if key in rows else "-"


def seconds(rows: dict, key: tuple) -> float:
    """Returns the median seconds of a row's pass.

    Args:
        rows: The rows.
        key: The row's key.

    Returns:
        The seconds.
    """
    mib, median, _ = rows[key]
    return mib / median


def single(rows: dict) -> str:
    """Returns the table of one long text in one call on one thread.

    Args:
        rows: The rows.

    Returns:
        The table.
    """
    body = []
    for name in ("gutenberg.txt", "wiki-ru.txt", "wiki-zh.txt", "enwik8"):
        for tokenizer in TOKENIZERS:
            body.append([
                tokenizer, f"`{name.removesuffix('.txt')}`",
                rate(rows, ("matisse", tokenizer, "encode", name)),
                rate(rows, ("matisse-py", tokenizer, "numpy fresh", name)),
                rate(rows, ("matisse-py", tokenizer, "encode fresh", name)),
                rate(rows, ("tiktoken", tokenizer, "numpy", name)),
                rate(rows, ("tiktoken", tokenizer, "encode", name)),
            ])
    header = ["Tokenizer", "Corpus", "C++ `encode()`", "Python array", "Python list", "tiktoken array",
              "tiktoken list"]
    return table(header, body, {2, 3, 4, 5, 6})


def costs(rows: dict) -> str:
    """Returns the table of where a Python call's time goes: the encoding, the conversion of the string and the list.

    Args:
        rows: The rows.

    Returns:
        The table.
    """
    body = []
    for name in ("gutenberg.txt", "wiki-ru.txt", "wiki-zh.txt", "enwik8"):
        for tokenizer in TOKENIZERS:
            cached = seconds(rows, ("matisse-py", tokenizer, "numpy cached", name))
            fresh = seconds(rows, ("matisse-py", tokenizer, "numpy fresh", name))
            listed = seconds(rows, ("matisse-py", tokenizer, "encode cached", name))
            count = rows[("matisse-py", tokenizer, "numpy cached", name)][2]
            body.append([
                tokenizer, f"`{name.removesuffix('.txt')}`", f"{cached * 1e3:.1f}", f"{(fresh - cached) * 1e3:.1f}",
                f"{(listed - cached) * 1e3:.1f}", f"{(listed - cached) / count * 1e9:.1f}",
            ])
    header = ["Tokenizer", "Corpus", "Encoding, ms", "Conversion, ms", "List, ms", "List, ns per id"]
    return table(header, body, {2, 3, 4, 5})


def prompts(latencies: dict) -> str:
    """Returns the table of one call per prompt: the mean microseconds of a call.

    Args:
        latencies: The latencies.

    Returns:
        The table.
    """
    body = []
    for name, shown in (("short.txt", "48 bytes at most"), ("prompts.txt", "100 to 2,000 bytes")):
        for tokenizer in TOKENIZERS:
            cells = [
                latencies.get(("matisse", tokenizer, "latency", name)),
                latencies.get(("matisse", tokenizer, "latency kept", name)),
                latencies.get(("matisse-py", tokenizer, "prompts latency", name)),
                latencies.get(("matisse-py", tokenizer, "prompts numpy latency", name)),
                latencies.get(("tiktoken", tokenizer, "prompts latency", name)),
                latencies.get(("tiktoken", tokenizer, "prompts numpy latency", name)),
            ]
            body.append([tokenizer, shown, *(f"{cell:.2f}" if cell is not None else "-" for cell in cells)])
    header = ["Tokenizer", "Prompts", "C++, own cache", "C++, kept cache", "Python list", "Python array",
              "tiktoken list", "tiktoken array"]
    return table(header, body, {2, 3, 4, 5, 6, 7})


def parallel(rows: dict) -> str:
    """Returns the table of one long text in parallel at each worker count.

    Args:
        rows: The rows.

    Returns:
        The table.
    """
    counts = [1, 2, 4, 8, 12, 16, 24]
    body = []
    for tokenizer in TOKENIZERS:
        body.append([tokenizer, "C++ `encode_parallel()`",
                     *(rate(rows, ("matisse", tokenizer, f"parallel x{count}", "enwik8")) for count in counts)])
        arrays = [rate(rows, ("matisse-py", tokenizer, f"parallel numpy x{count}", "enwik8")) for count in counts]
        body.append([tokenizer, "Python array", *arrays])
        body.append([tokenizer, "Python list", *(["-"] * (len(counts) - 1)),
                     rate(rows, ("matisse-py", tokenizer, "parallel list x24", "enwik8"))])
    header = ["Tokenizer", "Call", *(f"{count}" for count in counts)]
    return table(header, body, set(range(2, 2 + len(counts))))


def batch(rows: dict) -> str:
    """Returns the table of the prompts as a batch and over a pool of Python threads, at one and at 24 threads.

    Args:
        rows: The rows.

    Returns:
        The table.
    """
    body = []
    name = "prompts.txt"
    for count in (1, 24):
        for tokenizer in TOKENIZERS:
            body.append([
                tokenizer, f"{count}",
                rate(rows, ("matisse", tokenizer, f"batch x{count}", name)),
                rate(rows, ("matisse-py", tokenizer, f"batch x{count}", name)),
                rate(rows, ("matisse-py", tokenizer, f"batch numpy x{count}", name)),
                rate(rows, ("tiktoken", tokenizer, f"batch x{count}", name)),
                rate(rows, ("matisse-py", tokenizer, f"threads x{count}", name)),
                rate(rows, ("tiktoken", tokenizer, f"threads x{count}", name)),
            ])
    header = ["Tokenizer", "Threads", "C++ batch", "Python list", "Python array", "tiktoken", "Python pool",
              "tiktoken pool"]
    return table(header, body, set(range(1, len(header))))


def load(loads: dict) -> str:
    """Returns the table of the seconds a fresh process takes to build an encoding and encode a word.

    Args:
        loads: The load medians.

    Returns:
        The table.
    """
    body = []
    for tokenizer in TOKENIZERS:
        cells = [
            loads.get(("matisse-py", tokenizer, "load shipped")),
            loads.get(("matisse-py", tokenizer, "load decided")),
            loads.get(("tiktoken", tokenizer, "load")),
        ]
        body.append([tokenizer, *(f"{cell:.3f}" if cell is not None else "-" for cell in cells)])
    header = ["Tokenizer", "Shipped report, s", "Report decided, s", "tiktoken, s"]
    return table(header, body, {1, 2, 3})


def ratio(numerator: float | None, denominator: float | None) -> str:
    """Returns one figure over another, or a dash when either is missing.

    Args:
        numerator: The first figure.
        denominator: The second.

    Returns:
        The ratio.
    """
    return f"{numerator / denominator:.1f}x" if numerator and denominator else "-"


def median_rate(rows: dict, key: tuple) -> float | None:
    """Returns a row's median MiB/s, or None when no row holds it.

    Args:
        rows: The rows.
        key: The row's key.

    Returns:
        The figure.
    """
    return rows[key][1] if key in rows else None


def json_exactness(ids: dict, differs: set) -> str:
    """Returns the table of both packages' ids of every input under each tokenizer.json: the count and digest, and
    whether every ids line of either package gives them.

    Args:
        ids: The ids lines' counts and digests.
        differs: The tokenizers and inputs whose ids differ.

    Returns:
        The table.
    """
    body = []
    for name, shown in CORPORA.items():
        for tokenizer in TOKENIZERS:
            ours = ids.get(("matisse-py", tokenizer, name), set())
            theirs = ids.get(("hf", tokenizer, name), set())
            same = len(ours) == 1 and ours == theirs and (tokenizer, name) not in differs
            count, digest = next(iter(ours)) if len(ours) == 1 else (0, "")
            body.append([shown, tokenizer, f"{count:,}", f"`{digest}`", "yes" if same else "no"])
    header = ["Corpus", "tokenizer.json", "Ids", "SHA-256 of the ids", "Equal"]
    return table(header, body, {2})


def json_single(rows: dict) -> str:
    """Returns the table of one long text in one call on one thread under each tokenizer.json.

    Args:
        rows: The rows.

    Returns:
        The table.
    """
    body = []
    for name in ("gutenberg.txt", "wiki-ru.txt", "wiki-zh.txt", "enwik8"):
        for tokenizer in TOKENIZERS:
            listed = median_rate(rows, ("matisse-py", tokenizer, "json list", name))
            theirs = [median_rate(rows, ("hf", tokenizer, label, name)) for label in ("encode", "encode fast")]
            body.append([
                tokenizer, CORPORA[name],
                rate(rows, ("matisse-py", tokenizer, "json numpy", name)),
                rate(rows, ("matisse-py", tokenizer, "json list", name)),
                rate(rows, ("hf", tokenizer, "encode", name)),
                rate(rows, ("hf", tokenizer, "encode fast", name)),
                ratio(listed, max(filter(None, theirs), default=None)),
            ])
    header = ["tokenizer.json", "Corpus", "Python array", "Python list", "`encode()`", "`encode_batch_fast()`",
              "List over the faster"]
    return table(header, body, {2, 3, 4, 5, 6})


def json_prompts(latencies: dict) -> str:
    """Returns the table of one call per prompt under each tokenizer.json: the mean microseconds of a call.

    Args:
        latencies: The latencies.

    Returns:
        The table.
    """
    body = []
    for name, shown in (("short.txt", "48 bytes at most"), ("prompts.txt", "100 to 2,000 bytes")):
        for tokenizer in TOKENIZERS:
            cells = [
                latencies.get(("matisse-py", tokenizer, "json prompts latency", name)),
                latencies.get(("matisse-py", tokenizer, "json prompts numpy latency", name)),
                latencies.get(("hf", tokenizer, "prompts latency", name)),
                latencies.get(("hf", tokenizer, "prompts fast latency", name)),
            ]
            faster = min(filter(None, cells[2:]), default=None)
            body.append([tokenizer, shown, *(f"{cell:.2f}" if cell is not None else "-" for cell in cells),
                         ratio(faster, cells[0])])
    header = ["tokenizer.json", "Prompts", "Python list", "Python array", "`encode()`", "`encode_batch_fast()`",
              "List over the faster"]
    return table(header, body, {2, 3, 4, 5, 6})


def json_batch(rows: dict) -> str:
    """Returns the table of the prompts as one batch at one and at 24 threads under each tokenizer.json.

    Args:
        rows: The rows.

    Returns:
        The table.
    """
    body = []
    name = "prompts.txt"
    for count in (1, 24):
        for tokenizer in TOKENIZERS:
            listed = median_rate(rows, ("matisse-py", tokenizer, f"json batch x{count}", name))
            theirs = [median_rate(rows, ("hf", tokenizer, f"{label} x{count}", name))
                      for label in ("batch", "batch fast")]
            body.append([
                tokenizer, f"{count}",
                rate(rows, ("matisse-py", tokenizer, f"json batch x{count}", name)),
                rate(rows, ("matisse-py", tokenizer, f"json batch numpy x{count}", name)),
                rate(rows, ("hf", tokenizer, f"batch x{count}", name)),
                rate(rows, ("hf", tokenizer, f"batch fast x{count}", name)),
                ratio(listed, max(filter(None, theirs), default=None)),
            ])
    header = ["tokenizer.json", "Threads", "Python list", "Python array", "`encode_batch()`", "`encode_batch_fast()`",
              "List over the faster"]
    return table(header, body, set(range(1, len(header))))


def main() -> None:
    """Prints every table of the outputs in the directory the command line names."""
    rows, latencies, loads, ids, differs = read(pathlib.Path(sys.argv[1]))
    if any(vendor == "hf" for vendor, *_ in ids):
        tables = (("exactness", json_exactness(ids, differs)), ("single", json_single(rows)),
                  ("prompts", json_prompts(latencies)), ("batch", json_batch(rows)))
    else:
        tables = (("single", single(rows)), ("costs", costs(rows)), ("prompts", prompts(latencies)),
                  ("parallel", parallel(rows)), ("batch", batch(rows)), ("load", load(loads)))
    for title, text in tables:
        print(f"<!-- {title} -->\n{text}\n")


main()
