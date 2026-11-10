"""The Unicode boundary: matisse's class tables are pinned to Unicode 15.0.0, and tiktoken's regex engine carries tables
of its own, so the two can segment a text holding a code point assigned a letter, a number or, under o200k, a mark
after 15.0.0. These tests hold matisse to the 15.0.0 segmentation at every code point 15.0.0 leaves unassigned, and
state where tiktoken 0.14.0 departs from it: at every letter and number assigned in 15.1 and 16.0, and every mark under
o200k, and at none assigned in 17.0, so its tables are those of 16.0. At every code point 15.0.0 assigns, the two agree.
"""

from __future__ import annotations

import os
import pathlib
import unicodedata

import pytest

ASSIGNED_AFTER_15 = [
    (0x0088F, 0x0088F, "17.0", "Lo"), (0x00897, 0x00897, "16.0", "Mn"), (0x00C5C, 0x00C5C, "17.0", "Lo"),
    (0x00CDC, 0x00CDC, "17.0", "Lo"), (0x01ACF, 0x01ADD, "17.0", "Mn"), (0x01AE0, 0x01AEB, "17.0", "Mn"),
    (0x01C89, 0x01C89, "16.0", "Lu"), (0x01C8A, 0x01C8A, "16.0", "Ll"), (0x0A7CB, 0x0A7CC, "16.0", "Lu"),
    (0x0A7CD, 0x0A7CD, "16.0", "Ll"), (0x0A7CE, 0x0A7CE, "17.0", "Lu"), (0x0A7CF, 0x0A7CF, "17.0", "Ll"),
    (0x0A7D2, 0x0A7D2, "17.0", "Lu"), (0x0A7D4, 0x0A7D4, "17.0", "Lu"), (0x0A7DA, 0x0A7DA, "16.0", "Lu"),
    (0x0A7DB, 0x0A7DB, "16.0", "Ll"), (0x0A7DC, 0x0A7DC, "16.0", "Lu"), (0x0A7F1, 0x0A7F1, "17.0", "Lm"),
    (0x105C0, 0x105F3, "16.0", "Lo"), (0x10940, 0x10959, "17.0", "Lo"), (0x10D40, 0x10D49, "16.0", "Nd"),
    (0x10D4A, 0x10D4D, "16.0", "Lo"), (0x10D4E, 0x10D4E, "16.0", "Lm"), (0x10D4F, 0x10D4F, "16.0", "Lo"),
    (0x10D50, 0x10D65, "16.0", "Lu"), (0x10D69, 0x10D6D, "16.0", "Mn"), (0x10D6F, 0x10D6F, "16.0", "Lm"),
    (0x10D70, 0x10D85, "16.0", "Ll"), (0x10EC2, 0x10EC4, "16.0", "Lo"), (0x10EC5, 0x10EC5, "17.0", "Lm"),
    (0x10EC6, 0x10EC7, "17.0", "Lo"), (0x10EFA, 0x10EFB, "17.0", "Mn"), (0x10EFC, 0x10EFC, "16.0", "Mn"),
    (0x11380, 0x11389, "16.0", "Lo"), (0x1138B, 0x1138B, "16.0", "Lo"), (0x1138E, 0x1138E, "16.0", "Lo"),
    (0x11390, 0x113B5, "16.0", "Lo"), (0x113B7, 0x113B7, "16.0", "Lo"), (0x113B8, 0x113BA, "16.0", "Mc"),
    (0x113BB, 0x113C0, "16.0", "Mn"), (0x113C2, 0x113C2, "16.0", "Mc"), (0x113C5, 0x113C5, "16.0", "Mc"),
    (0x113C7, 0x113CA, "16.0", "Mc"), (0x113CC, 0x113CD, "16.0", "Mc"), (0x113CE, 0x113CE, "16.0", "Mn"),
    (0x113CF, 0x113CF, "16.0", "Mc"), (0x113D0, 0x113D0, "16.0", "Mn"), (0x113D1, 0x113D1, "16.0", "Lo"),
    (0x113D2, 0x113D2, "16.0", "Mn"), (0x113D3, 0x113D3, "16.0", "Lo"), (0x113E1, 0x113E2, "16.0", "Mn"),
    (0x116D0, 0x116E3, "16.0", "Nd"), (0x11B60, 0x11B60, "17.0", "Mn"), (0x11B61, 0x11B61, "17.0", "Mc"),
    (0x11B62, 0x11B64, "17.0", "Mn"), (0x11B65, 0x11B65, "17.0", "Mc"), (0x11B66, 0x11B66, "17.0", "Mn"),
    (0x11B67, 0x11B67, "17.0", "Mc"), (0x11BC0, 0x11BE0, "16.0", "Lo"), (0x11BF0, 0x11BF9, "16.0", "Nd"),
    (0x11DB0, 0x11DD8, "17.0", "Lo"), (0x11DD9, 0x11DD9, "17.0", "Lm"), (0x11DDA, 0x11DDB, "17.0", "Lo"),
    (0x11DE0, 0x11DE9, "17.0", "Nd"), (0x11F5A, 0x11F5A, "16.0", "Mn"), (0x13460, 0x143FA, "16.0", "Lo"),
    (0x16100, 0x1611D, "16.0", "Lo"), (0x1611E, 0x16129, "16.0", "Mn"), (0x1612A, 0x1612C, "16.0", "Mc"),
    (0x1612D, 0x1612F, "16.0", "Mn"), (0x16130, 0x16139, "16.0", "Nd"), (0x16D40, 0x16D42, "16.0", "Lm"),
    (0x16D43, 0x16D6A, "16.0", "Lo"), (0x16D6B, 0x16D6C, "16.0", "Lm"), (0x16D70, 0x16D79, "16.0", "Nd"),
    (0x16EA0, 0x16EB8, "17.0", "Lu"), (0x16EBB, 0x16ED3, "17.0", "Ll"), (0x16FF2, 0x16FF3, "17.0", "Lm"),
    (0x16FF4, 0x16FF6, "17.0", "Nl"), (0x187F8, 0x187FF, "17.0", "Lo"), (0x18CFF, 0x18CFF, "16.0", "Lo"),
    (0x18D09, 0x18D1E, "17.0", "Lo"), (0x18D80, 0x18DF2, "17.0", "Lo"), (0x1CCF0, 0x1CCF9, "16.0", "Nd"),
    (0x1E5D0, 0x1E5ED, "16.0", "Lo"), (0x1E5EE, 0x1E5EF, "16.0", "Mn"), (0x1E5F0, 0x1E5F0, "16.0", "Lo"),
    (0x1E5F1, 0x1E5FA, "16.0", "Nd"), (0x1E6C0, 0x1E6DE, "17.0", "Lo"), (0x1E6E0, 0x1E6E2, "17.0", "Lo"),
    (0x1E6E3, 0x1E6E3, "17.0", "Mn"), (0x1E6E4, 0x1E6E5, "17.0", "Lo"), (0x1E6E6, 0x1E6E6, "17.0", "Mn"),
    (0x1E6E7, 0x1E6ED, "17.0", "Lo"), (0x1E6EE, 0x1E6EF, "17.0", "Mn"), (0x1E6F0, 0x1E6F4, "17.0", "Lo"),
    (0x1E6F5, 0x1E6F5, "17.0", "Mn"), (0x1E6FE, 0x1E6FE, "17.0", "Lo"), (0x1E6FF, 0x1E6FF, "17.0", "Lm"),
    (0x2B73A, 0x2B73F, "17.0", "Lo"), (0x2CEA2, 0x2CEAD, "17.0", "Lo"), (0x2EBF0, 0x2EE5D, "16.0", "Lo"),
    (0x323B0, 0x33479, "17.0", "Lo"),
]
"""Every code point Unicode 15.0.0 leaves unassigned that a later version assigns a letter, a number or a mark, with the
first of the data files of 16.0.0 and 17.0.0 that assigns it, 16.0 standing for 15.1 and 16.0, and its general category
there, as derive_assigned_after_15() derives them from the two UnicodeData.txt files and Python 3.12's unicodedata,
whose version is 15.0.0. 9,754 code points: 5,055 by 16.0.0 and 4,699 more by 17.0.0. Beside them one code point
assigned in 15.0.0 changes a class the patterns read, U+0295, from Ll to Lo in 17.0.0."""

PLANES = [*range(0x0000, 0xD800), *range(0xE000, 0x40000), *range(0xE0000, 0xF0000)]
"""The scalar values of the planes Unicode assigns letters, numbers and marks in, 0 to 3, and 14."""


def after_15() -> dict[int, tuple[str, str]]:
    """Returns every code point of ASSIGNED_AFTER_15 with its version and category.

    Returns:
        The version and the category by code point.
    """
    return {
        point: (version, category)
        for first, last, version, category in ASSIGNED_AFTER_15
        for point in range(first, last + 1)
    }


def unassigned_in_15() -> list[int]:
    """Returns the code points of the planes that Unicode 15.0.0 leaves unassigned: those the running Python's
    unicodedata leaves unassigned, and those of ASSIGNED_AFTER_15 where its version is later.

    Returns:
        The code points, ascending.
    """
    later = after_15()
    return [point for point in PLANES if point in later or unicodedata.category(chr(point)) == "Cn"]


def departs(pattern_reads_marks: bool) -> set[int]:
    """Returns where an engine with Unicode 16.0.0's tables departs from 15.0.0's on the probe: every letter and number
    assigned by 16.0.0, and every mark where the pattern reads marks.

    Args:
        pattern_reads_marks: Whether the pattern reads marks, as o200k's does.

    Returns:
        The code points.
    """
    classes = ("L", "N", "M") if pattern_reads_marks else ("L", "N")
    return {point for point, (version, category) in after_15().items() if version == "16.0" and category[0] in classes}


def probes(points: list[int]) -> list[str]:
    """Returns the probe of each code point, the code point followed by `'s`.

    Every pattern reads a code point of the other class there as one piece with the apostrophe and then `s`, and a
    letter or a number as a piece of its own before the contraction, so the ids tell the two apart.

    Args:
        points: The code points.

    Returns:
        The probes.
    """
    return [chr(point) + "'s" for point in points]


def test_the_ranges_are_ordered_and_hold_the_counts_the_data_files_give():
    points = after_15()
    assert len(points) == 9754
    assert sum(version == "16.0" for version, _ in points.values()) == 5055
    starts = [first for first, _, _, _ in ASSIGNED_AFTER_15]
    assert starts == sorted(starts)


def test_matisse_segments_every_code_point_15_leaves_unassigned_as_the_other_class(pair):
    ours, theirs = pair
    points = unassigned_in_15()
    # The 15.0.0 segmentation of the probe: the code point and the apostrophe as one piece, then `s`.
    apostrophe = [theirs._encode_single_piece((chr(point) + "'").encode()) for point in points]
    letter = theirs._encode_single_piece(b"s")
    expected = [ids + letter for ids in apostrophe]
    encoded = ours.encode_ordinary_batch(probes(points))
    wrong = [hex(point) for point, ids, want in zip(points, encoded, expected) if ids != want]
    assert wrong == []


def test_tiktoken_departs_from_15_exactly_at_the_letters_numbers_and_marks_of_15_1_and_16(pair):
    ours, theirs = pair
    points = unassigned_in_15()
    expected = ours.encode_ordinary_batch(probes(points))
    encoded = theirs.encode_ordinary_batch(probes(points))
    departed = {point for point, ids, want in zip(points, encoded, expected) if ids != want}
    assert departed == departs(pattern_reads_marks=ours.name == "o200k_base")


def test_matisse_and_tiktoken_agree_at_every_code_point_15_assigns(pair):
    ours, theirs = pair
    later = after_15()
    points = [point for point in PLANES if point not in later and unicodedata.category(chr(point)) != "Cn"]
    texts = [text for point in points for text in (chr(point) + "'s", "x" + chr(point) + "1", " " + chr(point) * 2)]
    assert ours.encode_ordinary_batch(texts) == theirs.encode_ordinary_batch(texts)


def derive_assigned_after_15(directory: pathlib.Path) -> list[tuple[int, int, str, str]]:
    """Derives ASSIGNED_AFTER_15 from the UnicodeData.txt of 16.0.0 and 17.0.0 and the running Python's unicodedata,
    which must be of 15.0.0, as Python 3.12's is.

    Args:
        directory: A directory holding 16.0.0/UnicodeData.txt and 17.0.0/UnicodeData.txt.

    Returns:
        The ranges, each a first and last code point, a version and a category.
    """
    def categories(path: pathlib.Path) -> dict[int, str]:
        found: dict[int, str] = {}
        first = 0
        for line in path.read_text().splitlines():
            fields = line.split(";")
            point = int(fields[0], 16)
            if fields[1].endswith("First>"):
                first = point
            elif fields[1].endswith("Last>"):
                found.update(dict.fromkeys(range(first, point + 1), fields[2]))
            else:
                found[point] = fields[2]
        return found

    sixteen = categories(directory / "16.0.0" / "UnicodeData.txt")
    seventeen = categories(directory / "17.0.0" / "UnicodeData.txt")
    ranges: list[list[object]] = []
    for point in range(0x110000):
        category = seventeen.get(point, "Cn")
        if 0xD800 <= point < 0xE000 or unicodedata.category(chr(point)) != "Cn" or category[0] not in "LNM":
            continue
        version = "16.0" if point in sixteen else "17.0"
        if ranges and ranges[-1][1] == point - 1 and ranges[-1][2:] == [version, category]:
            ranges[-1][1] = point
        else:
            ranges.append([point, point, version, category])
    return [tuple(each) for each in ranges]


@pytest.mark.skipif(not os.environ.get("MATISSE_UCD_DIR"), reason="no Unicode data: set MATISSE_UCD_DIR")
@pytest.mark.skipif(unicodedata.unidata_version != "15.0.0", reason="the derivation needs Python's 15.0.0 tables")
def test_the_ranges_are_those_the_unicode_data_files_derive():
    assert derive_assigned_after_15(pathlib.Path(os.environ["MATISSE_UCD_DIR"])) == ASSIGNED_AFTER_15
