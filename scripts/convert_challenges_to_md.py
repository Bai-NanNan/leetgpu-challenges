#!/usr/bin/env python3
"""Convert every challenge HTML prompt to Markdown beside its source file."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pypandoc


MATH_PATTERN = re.compile(
    r"(?P<display_dollars>\$\$(?P<dollars>.*?)\$\$)"
    r"|(?P<inline>\\\((?P<inline_content>.*?)\\\))"
    r"|(?P<display_brackets>\\\[(?P<brackets>.*?)\\\])",
    re.DOTALL,
)


def _protect_math(source: str) -> tuple[str, dict[str, str]]:
    """Keep MathJax delimiters out of Pandoc's HTML reader."""

    replacements: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"LEETGPUMATH{len(replacements):06d}MARKER"
        if match.group("inline") is not None:
            replacements[token] = f"${match.group('inline_content').strip()}$"
        else:
            content = match.group("dollars") or match.group("brackets") or ""
            replacements[token] = f"\n\n$$\n{content.strip()}\n$$\n\n"
        return token

    return MATH_PATTERN.sub(replace, source), replacements


def _restore_math(markdown: str, replacements: dict[str, str]) -> str:
    for token, formula in replacements.items():
        markdown = markdown.replace(token, formula)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"


def convert_html(source: Path, destination: Path) -> None:
    protected_html, replacements = _protect_math(source.read_text())
    markdown = pypandoc.convert_text(
        protected_html,
        to="gfm",
        format="html",
        extra_args=["--wrap=none"],
    )
    destination.write_text(_restore_math(markdown, replacements))


def convert_all(challenges_dir: Path, dry_run: bool) -> tuple[int, int]:
    converted = 0
    failed = 0
    for source in sorted(challenges_dir.glob("**/challenge.html")):
        destination = source.with_suffix(".md")
        relative_source = source.relative_to(challenges_dir.parent)
        if dry_run:
            print(f"Would convert {relative_source} -> {destination.name}")
            converted += 1
            continue

        try:
            convert_html(source, destination)
        except (OSError, RuntimeError) as exc:
            print(f"Failed to convert {relative_source}: {exc}", file=sys.stderr)
            failed += 1
            continue

        print(f"Converted {relative_source} -> {destination.name}")
        converted += 1
    return converted, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert challenges/**/challenge.html files to GitHub-Flavored Markdown."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List the files that would be converted without writing them."
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    converted, failed = convert_all(repository_root / "challenges", args.dry_run)
    print(f"Summary: {converted} converted, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
