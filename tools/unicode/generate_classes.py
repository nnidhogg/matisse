#!/usr/bin/env python3
"""Generates the code point tables behind a pattern's class alphabet from the Unicode Character Database, one table for
each set the tokenizer's pattern names.

Each set is named on the command line as TABLE=SET: TABLE is the name of the array in the output, and SET is a general
category of one letter, the union of its subcategories, such as L for Lu, Ll, Lt, Lm and Lo, a general category of two
letters, such as Lu, or a property of PropList.txt, such as White_Space, which is what the regex crate and Python's re
read \\s as. Reads PropList.txt and extracted/DerivedGeneralCategory.txt, published at
https://www.unicode.org/Public/<version>/ucd/, and emits one table file for the database it is given, the tables in the
order named. The pattern reads each set as itself, and a code point is classed by the first set holding it, so a code
point in two of them would be read as in the first alone: the sets named must be disjoint, and a database where two of
them share a code point is refused. The generated file is checked in, so building matisse never touches the network;
rerun this script against a newer database to move the pinned Unicode version.

Each tokenizer library keeps its own output, libs/<name>/src/class_ranges.inc, with the sets its pattern names.
GPT-2's, cl100k's and Llama 3's patterns name \\p{L}, \\p{N} and \\s, and every table is pinned to Unicode 15.0.0, the
version of the Python reference the fixtures are written by, so that the fixtures replay:

    tools/unicode/generate_classes.py --version 15.0.0 PropList.txt DerivedGeneralCategory.txt \\
        libs/gpt2/src/class_ranges.inc letter_ranges=L number_ranges=N white_space_ranges=White_Space

and the same for libs/cl100k/src/class_ranges.inc and libs/llama3/src/class_ranges.inc. o200k's pattern names \\p{Lu},
\\p{Ll}, \\p{Lt}, \\p{Lm}, \\p{Lo}, \\p{M}, \\p{N} and \\s:

    tools/unicode/generate_classes.py --version 15.0.0 PropList.txt DerivedGeneralCategory.txt \\
        libs/o200k/src/class_ranges.inc upper_ranges=Lu lower_ranges=Ll title_ranges=Lt modifier_ranges=Lm \\
        other_letter_ranges=Lo mark_ranges=M number_ranges=N white_space_ranges=White_Space

Pass --version to require the database to be a given version.

Usage: tools/unicode/generate_classes.py [--version X.Y.Z] <PropList.txt> <DerivedGeneralCategory.txt> <output.inc>
       TABLE=SET...
"""

import hashlib
import re
import sys
from pathlib import Path


def parse_ranges(lines: list[str], property_name: str) -> list[tuple[int, int]]:
    """Extracts the code point ranges holding the given property or general category, unmerged."""
    pattern = re.compile(r"^([0-9A-F]+)(?:\.\.([0-9A-F]+))?\s*;\s*" + property_name + r"\s*#")

    ranges = []

    for line in lines:
        if match := pattern.match(line):
            first = int(match.group(1), 16)
            last = int(match.group(2), 16) if match.group(2) else first

            if not first <= last <= 0x10FFFF:
                raise ValueError(f"invalid range {first:X}..{last:X}")

            ranges.append((first, last))

    if not ranges:
        raise ValueError(f"no range holds {property_name}")

    return ranges


def merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorts and merges ranges, adjacent ones included, since a union of categories lists them apart."""
    ranges = sorted(ranges)

    merged = [ranges[0]]

    for first, last in ranges[1:]:
        if first <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))

    return merged


def emit_array(name: str, ranges: list[tuple[int, int]]) -> str:
    entries = "\n".join(f"        {{.first = 0x{first:X}, .last = 0x{last:X}}}," for first, last in ranges)

    return (
        f"constexpr std::array<Code_point_range, {len(ranges)}> {name}{{{{\n{entries}\n}}}};\n"
    )


def version_of(lines: list[str], stem: str) -> str:
    match = re.match(r"# " + stem + r"-(\d+\.\d+\.\d+)\.txt", lines[0])

    if not match:
        raise ValueError(f"first line does not name the {stem} database version")

    return match.group(1)


def table_of(argument: str, props: list[str], categories: list[str]) -> tuple[str, str, list[tuple[int, int]]]:
    """Reads one TABLE=SET argument: the table's name, the set's name and the set's ranges, merged. A set named by one
    capital letter is the union of the general categories that letter begins, a set named by a capital and a small
    letter is that general category, and a set of any other name is a property of PropList.txt."""
    table, separator, name = argument.partition("=")

    if not separator or not table or not name:
        sys.exit(__doc__)

    if re.fullmatch(r"[A-Z]", name):
        return table, name, merge(parse_ranges(categories, name + "[a-z]"))

    if re.fullmatch(r"[A-Z][a-z]", name):
        return table, name, merge(parse_ranges(categories, name))

    return table, name, merge(parse_ranges(props, name))


def main() -> None:
    arguments = sys.argv[1:]

    required = None

    if arguments and arguments[0] == "--version":
        if len(arguments) < 2:
            sys.exit(__doc__)

        required, arguments = arguments[1], arguments[2:]

    if len(arguments) < 4:
        sys.exit(__doc__)

    raw = [Path(argument).read_bytes() for argument in arguments[:2]]

    props, categories = (text.decode("utf-8").splitlines() for text in raw)

    output = Path(arguments[2])

    versions = {version_of(props, "PropList"), version_of(categories, "DerivedGeneralCategory")}

    if len(versions) != 1:
        raise ValueError(f"the two files name different database versions: {sorted(versions)}")

    version = versions.pop()

    if required is not None and version != required:
        raise ValueError(f"the database is Unicode {version}, and {required} was asked for")

    tables = [table_of(argument, props, categories) for argument in arguments[3:]]

    # A code point is classed by the first set holding it, so two sets sharing one would be read as the first alone.
    def shared(ranges: list[tuple[int, int]], other: list[tuple[int, int]]) -> int | None:
        points = (point for first, last in ranges for point in range(first, last + 1))

        return next((point for point in points if any(low <= point <= high for low, high in other)), None)

    for index, (_, name, ranges) in enumerate(tables):
        for _, other, other_ranges in tables[index + 1:]:
            if (point := shared(ranges, other_ranges)) is not None:
                raise ValueError(f"U+{point:04X} is in {name} and in {other}")

    names = ["PropList.txt", "extracted/DerivedGeneralCategory.txt"]

    sources = "\n".join(
        f"// Source: https://www.unicode.org/Public/{version}/ucd/{name}\n"
        f"// Source SHA-256: {hashlib.sha256(bytes_).hexdigest()}"
        for name, bytes_ in zip(names, raw)
    )

    header = (
        f"// Generated by tools/unicode/generate_classes.py from the Unicode {version} database; do not edit.\n"
        f"{sources}\n"
        "// The ranges derive from the Unicode Character Database, (c) Unicode, Inc., used under the Unicode\n"
        "// License v3; the complete notice is in THIRD_PARTY_NOTICES.md at the repository root.\n"
        "\n"
        f'constexpr std::string_view class_unicode_version{{"{version}"}};\n'
        "\n"
    )

    output.write_text(
        header + "\n".join(emit_array(table, ranges) for table, _, ranges in tables),
        encoding="utf-8",
        newline="\n",
    )

    def points(ranges: list[tuple[int, int]]) -> int:
        return sum(last - first + 1 for first, last in ranges)

    counts = ", ".join(f"{name} {len(ranges)} ranges ({points(ranges)} code points)" for _, name, ranges in tables)

    print(f"Unicode {version}: {counts} -> {output}")


if __name__ == "__main__":
    main()
