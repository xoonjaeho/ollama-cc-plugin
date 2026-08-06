#!/usr/bin/env python3
"""Self-check for tool_read_file absolute line numbering."""

import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama_agent


def _first_line_number(result):
    first = result.splitlines()[0]
    m = re.match(r"\s*(\d+):", first)
    assert m, "no line number on first line: %r" % first
    return int(m.group(1))


def _parse_footer(result):
    m = re.search(r"\[truncated at \d+ bytes; call read_file again with offset=(\d+) for more\]", result)
    assert m, "truncation footer missing"
    return int(m.group(1))


def test_numbered_read():
    """Verify absolute line numbers, byte accounting, and auto-advance correctness."""
    original_cap = ollama_agent.READ_CAP
    cap = 64
    ollama_agent.READ_CAP = cap
    try:
        lines = ["line %03d こんにちは padding to grow\n" % i for i in range(50)]
        data = "".join(lines).encode("utf-8")
        assert len(data) > cap * 3, "fixture too small"

        with tempfile.TemporaryDirectory() as td:
            path = "demo.txt"
            full = os.path.join(td, path)
            with open(full, "wb") as f:
                f.write(data)

            filesize = len(data)

            # (a) byte-accounting invariant at offset 0
            served0 = 0
            result0 = ollama_agent.tool_read_file(td, {"path": path, "offset": served0})
            next0 = _parse_footer(result0)
            assert next0 - served0 == cap, (
                "offset-0 byte invariant broken: next=%d served=%d cap=%d" % (next0, served0, cap)
            )
            assert _first_line_number(result0) == 1

            # (b) auto-advance: dispatcher rewrites offset to served_to.
            eff_off = ollama_agent._next_read_offset(0, cap, filesize)
            assert eff_off == cap, "auto-advance did not move to cap"
            result_auto = ollama_agent.tool_read_file(td, {"path": path, "offset": eff_off})

            true_start = data[:eff_off].count(b"\n") + 1
            got_start = _first_line_number(result_auto)
            assert got_start == true_start, (
                "auto-advance first line wrong: got %d want %d" % (got_start, true_start)
            )

            # (a) byte invariant in auto-advance case
            next_auto = _parse_footer(result_auto)
            assert next_auto - eff_off == cap, (
                "auto-advance byte invariant broken: next=%d served=%d cap=%d" % (next_auto, eff_off, cap)
            )

            # (c) footer still advertises offset + READ_CAP
            assert next0 == cap, "offset-0 footer advertises wrong next offset"
            assert next_auto == cap * 2, "auto-advance footer advertises wrong next offset"

            # Boundary line is marked partial when the byte cap cuts mid-line.
            body_lines = [ln for ln in result0.splitlines() if not ln.startswith("[")]
            assert body_lines[-1].endswith(" [partial]"), "missing partial marker on boundary line"
    finally:
        ollama_agent.READ_CAP = original_cap


if __name__ == "__main__":
    test_numbered_read()
    print("ok")
