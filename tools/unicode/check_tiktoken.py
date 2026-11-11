#!/usr/bin/env python3
"""Holds matisse's code point classes to tiktoken's, whose regex engine carries Unicode tables of its own, so that a
move of tiktoken's tables past the Unicode version matisse's are pinned to is caught. Both checks run over every Unicode
scalar value under GPT-2's, cl100k's, o200k's and Llama 3's patterns.

The tables. A code point that the tables of a tokenizer's src/class_ranges.inc leave out of every set they hold is of
the other class, and tiktoken must read it so: the code point followed by `'s` is then one piece with the apostrophe and
then `s`, where a letter, a number or, under o200k's pattern, a mark is a piece before the contraction. A code point
tiktoken reads otherwise is one its tables assign a class the pattern names and matisse's do not, which is what tables
of a later Unicode version look like. This check needs tiktoken and the rank files alone.

The ids. Every scalar value in each of five contexts, `C's`, `xC1`, ` CC`, `aCA` and `ACa`, one text per context with
the code points in ascending order, and every corpus named, are encoded by matisse's corpus probe and by tiktoken's
encode_ordinary(), and the two are compared by the count and the SHA-256 of their ids, the probe's ids lines. This
check needs a build's matisse_corpus as well, and the directory to hold all five pinned files, since the probe reads all
four tokenizers.

The rank files are read from VOCAB_DIR, each held to the SHA-256 tiktoken pins, and served to tiktoken's own
openai_public constructors through TIKTOKEN_CACHE_DIR, so that nothing is fetched; Llama 3's encoding is built from
llama3.tiktoken with the pattern of Meta's tokenizer.py, and the first check skips a tokenizer whose files the directory
lacks. Run it with the newest tiktoken, so that a release of it with tables of a later Unicode version is held to the
pin. Any difference exits 1.

Usage: tools/unicode/check_tiktoken.py VOCAB_DIR [MATISSE_CORPUS [CORPUS...]]
"""

import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PINNED = {
    "gpt2-vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    "gpt2-merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "cl100k_base.tiktoken": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    "o200k_base.tiktoken": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    "llama3.tiktoken": "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
}
"""The SHA-256 of each pinned file, as design.md gives them."""

URLS = {
    "gpt2-vocab.json": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
    "gpt2-merges.txt": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
    "cl100k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "o200k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
}
"""The URL tiktoken's openai_public reads each file from, which names the file's entry in its cache."""

FILES = {
    "gpt2": ["gpt2-vocab.json", "gpt2-merges.txt"],
    "cl100k": ["cl100k_base.tiktoken"],
    "o200k": ["o200k_base.tiktoken"],
    "llama3": ["llama3.tiktoken"],
}
"""The pinned files each tokenizer is read from, by the name matisse's libraries and probe give it."""

LLAMA3_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
                  r"\s*[\r\n]+|\s+(?!\S)|\s+")
"""Llama 3's pattern, as Meta's tokenizer.py gives it."""

CONTEXTS = {
    "contraction": lambda point: point + "'s",
    "between": lambda point: "x" + point + "1",
    "doubled": lambda point: " " + point + point,
    "lower-upper": lambda point: "a" + point + "A",
    "upper-lower": lambda point: "A" + point + "a",
}
"""The contexts the second check writes every code point in, by the name of the text that holds them."""

SCALARS = [*range(0xD800), *range(0xE000, 0x110000)]
"""Every Unicode scalar value, the code points but the surrogates."""


def encodings(vocab: Path, names: list[str]) -> dict:
    """Builds tiktoken's encoding of each tokenizer named from the pinned files, served to tiktoken's own constructors
    through a cache directory of its own, so that nothing is fetched.

    Args:
        vocab: The directory holding the pinned files.
        names: The tokenizers, each of FILES.

    Returns:
        The encoding by tokenizer.
    """
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe
    from tiktoken_ext import openai_public

    def llama3() -> dict:
        ranks = load_tiktoken_bpe(str(vocab / "llama3.tiktoken"), PINNED["llama3.tiktoken"])

        return {"name": "llama3", "pat_str": LLAMA3_PATTERN, "mergeable_ranks": ranks, "special_tokens": {}}

    constructors = {"gpt2": openai_public.gpt2, "cl100k": openai_public.cl100k_base, "o200k": openai_public.o200k_base,
                    "llama3": llama3}

    with tempfile.TemporaryDirectory(prefix="matisse-tiktoken-") as cache:
        for name in names:
            for file in FILES[name]:
                data = (vocab / file).read_bytes()

                if hashlib.sha256(data).hexdigest() != PINNED[file]:
                    sys.exit(f"{vocab / file} is not the pinned file")

                if file in URLS:
                    shutil.copyfile(vocab / file, Path(cache) / hashlib.sha1(URLS[file].encode()).hexdigest())

        os.environ["TIKTOKEN_CACHE_DIR"] = cache

        return {name: tiktoken.Encoding(**constructors[name]()) for name in names}


def classed(name: str) -> tuple[str, set[int]]:
    """Reads a tokenizer's class tables: the Unicode version they are generated from, and every code point a set of them
    holds.

    Args:
        name: The tokenizer, the name of its library under libs/.

    Returns:
        The version and the code points.
    """
    text = (ROOT / "libs" / name / "src" / "class_ranges.inc").read_text(encoding="utf-8")

    version = re.search(r'class_unicode_version\{"([0-9.]+)"\}', text).group(1)

    ranges = re.findall(r"\.first = 0x([0-9A-F]+), \.last = 0x([0-9A-F]+)", text)

    points = {point for first, last in ranges for point in range(int(first, 16), int(last, 16) + 1)}

    return version, points


def check_tables(name: str, encoding) -> int:
    """Holds tiktoken to a tokenizer's tables at every scalar value the tables leave out of every set: tiktoken must
    read the code point followed by `'s` as the code point and the apostrophe, one piece, and then `s`.

    Args:
        name: The tokenizer.
        encoding: tiktoken's encoding of it.

    Returns:
        The number of code points tiktoken reads otherwise.
    """
    version, points = classed(name)

    others = [point for point in SCALARS if point not in points]

    probes = encoding.encode_ordinary_batch([chr(point) + "'s" for point in others])

    pieces = encoding.encode_ordinary_batch([chr(point) + "'" for point in others])

    letter = encoding.encode_ordinary("s")

    departed = [point for point, ids, piece in zip(others, probes, pieces) if ids != piece + letter]

    shown = " ".join(f"U+{point:04X}" for point in departed[:20])

    print(f"tables {name}: Unicode {version}, {len(others)} code points of the other class, {len(departed)} read "
          f"otherwise by tiktoken{': ' + shown if departed else ''}{' ...' if len(departed) > 20 else ''}")

    return len(departed)


def digest_of(ids: list[int]) -> str:
    """Returns the SHA-256 of a run of ids written as little-endian 32-bit words, the fixtures' digest of ids."""
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def check_ids(probe: Path, vocab: Path, built: dict, corpora: list[Path]) -> int:
    """Encodes the context texts and the corpora with matisse's corpus probe and with tiktoken, and compares the two by
    the count and the digest of their ids.

    Args:
        probe: The build's matisse_corpus.
        vocab: The directory holding the pinned files.
        built: tiktoken's encoding by tokenizer.
        corpora: The corpora named on the command line.

    Returns:
        The number of differences, one for each text and tokenizer whose ids differ and one for each probe run that
        fails.
    """
    differing = 0

    with tempfile.TemporaryDirectory(prefix="matisse-contexts-") as directory:
        texts = {}

        for context, write in CONTEXTS.items():
            path = Path(directory) / f"every-code-point-{context}.txt"

            texts[path] = "".join(write(chr(point)) for point in SCALARS)

            path.write_text(texts[path], encoding="utf-8", newline="")

        texts |= {path: path.read_bytes().decode("utf-8") for path in corpora}

        found = {}

        # The probe runs once per text, and a failure names its text.
        for path in texts:
            run = subprocess.run([str(probe), str(vocab), str(path)], capture_output=True, text=True, check=False)

            for line in run.stdout.splitlines():
                if match := re.fullmatch(r"ids (\S+) (\S+) (\d+) ([0-9a-f]{64})", line):
                    found[(match.group(1), match.group(2))] = (int(match.group(3)), match.group(4))

            if run.returncode != 0:
                print(f"ids {path.name}: the corpus probe failed with {run.returncode}: {run.stderr.strip()}")

                differing += 1

    for path, text in texts.items():
        label = path.name.removesuffix(".txt")

        for name, encoding in built.items():
            ids = encoding.encode_ordinary(text)

            theirs = (len(ids), digest_of(ids))

            ours = found.get((name, label))

            verdict = "equal" if ours == theirs else f"differ, matisse {ours}"

            differing += ours != theirs

            print(f"ids {name} {label} {theirs[0]} {theirs[1]}: {verdict}")

    return differing


def main() -> None:
    arguments = sys.argv[1:]

    if not arguments:
        sys.exit(__doc__)

    vocab = Path(arguments[0])

    names = [name for name in FILES if all((vocab / file).exists() for file in FILES[name])]

    import tiktoken

    print(f"tiktoken {tiktoken.__version__}, tokenizers {', '.join(names)}")

    built = encodings(vocab, names)

    failures = sum(check_tables(name, encoding) for name, encoding in built.items())

    for name in FILES:
        if name not in built:
            print(f"tables {name}: skipped, {vocab} lacks {' or '.join(FILES[name])}")

    if len(arguments) > 1:
        failures += check_ids(Path(arguments[1]), vocab, built, [Path(argument) for argument in arguments[2:]])

    print("OK" if failures == 0 else "FAILED")

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
