#!/usr/bin/env python3
"""Generates the code point tables behind the class alphabets of the GPT-2 and cl100k patterns from the Unicode Character
Database.

The patterns name three sets, which their engines read from Unicode: \\p{L}, the general category L (Lu, Ll, Lt, Lm and
Lo), \\p{N}, the general category N (Nd, Nl and No), and \\s, the White_Space property, which is what the regex crate and
Python's reference read it as. Reads PropList.txt and extracted/DerivedGeneralCategory.txt, published at
https://www.unicode.org/Public/<version>/ucd/, and emits one table file for the database it is given. The generated file
is checked in, so building matisse never touches the network; rerun this script against a newer database to move the
pinned Unicode version.

libs/pretok/src/class_ranges.inc, the default output, is pinned to Unicode 15.0.0, the version of the Python reference
the fixtures are written by, so that the fixtures replay. Pass --version to require the database to be a given version.

Usage: tools/unicode/generate_classes.py [--version X.Y.Z] <PropList.txt> <DerivedGeneralCategory.txt> [output.inc]
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


def main() -> None:
    arguments = sys.argv[1:]

    required = None

    if arguments and arguments[0] == "--version":
        if len(arguments) < 2:
            sys.exit(__doc__)

        required, arguments = arguments[1], arguments[2:]

    if len(arguments) < 2:
        sys.exit(__doc__)

    raw = [Path(argument).read_bytes() for argument in arguments[:2]]

    props, categories = (text.decode("utf-8").splitlines() for text in raw)

    output = Path(arguments[2] if len(arguments) > 2 else Path(__file__).parents[2] / "libs/pretok/src/class_ranges.inc")

    versions = {version_of(props, "PropList"), version_of(categories, "DerivedGeneralCategory")}

    if len(versions) != 1:
        raise ValueError(f"the two files name different database versions: {sorted(versions)}")

    version = versions.pop()

    if required is not None and version != required:
        raise ValueError(f"the database is Unicode {version}, and {required} was asked for")

    letter = merge([r for category in ("Lu", "Ll", "Lt", "Lm", "Lo") for r in parse_ranges(categories, category)])
    number = merge([r for category in ("Nd", "Nl", "No") for r in parse_ranges(categories, category)])
    white_space = merge(parse_ranges(props, "White_Space"))

    # The classes are read in the order white space, letter, number, so a code point in two sets would be classed by
    # the first; the three are disjoint in every version, and a database where they are not is refused.
    def holds(ranges: list[tuple[int, int]], point: int) -> bool:
        return any(first <= point <= last for first, last in ranges)

    for first, last in white_space:
        for point in range(first, last + 1):
            if holds(letter, point) or holds(number, point):
                raise ValueError(f"U+{point:04X} is white space and a letter or a number")

    for first, last in number:
        for point in range(first, last + 1):
            if holds(letter, point):
                raise ValueError(f"U+{point:04X} is a letter and a number")

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
        header + emit_array("letter_ranges", letter) + "\n" + emit_array("number_ranges", number) + "\n"
        + emit_array("white_space_ranges", white_space),
        encoding="utf-8",
        newline="\n",
    )

    def points(ranges: list[tuple[int, int]]) -> int:
        return sum(last - first + 1 for first, last in ranges)

    print(f"Unicode {version}: L {len(letter)} ranges ({points(letter)} code points), "
          f"N {len(number)} ranges ({points(number)} code points), "
          f"White_Space {len(white_space)} ranges ({points(white_space)} code points) -> {output}")


if __name__ == "__main__":
    main()
