"""Phase 12 — markdown → HTML → PDF. One script, both documents.

    python -m scripts.build_pdf docs/reports/ALP-System-Report.md \
        --css docs/reports/report.css --out docs/reports/ALP-System-Report.pdf

Renders with `markdown` (tables, footnotes, fenced code, attribute lists) and
prints with `weasyprint`, which embeds the figures cleanly and honours the CSS
paged-media rules the stylesheet uses for the running header and page numbers.
Pandoc is the documented fallback and is not required.

Two things this does beyond a plain conversion, both because the documents are
full of figures and numbers that must stay traceable:

* **Figure references are resolved and checked.** An `![alt](path)` whose file
  is missing is a build failure, not a broken image in a PDF nobody opens.
* **`{{artifact:path:json.pointer}}` is substituted from the artifact tree**, so
  a number in the prose is read out of `artifacts/` at build time rather than
  typed. `{{artifact:artifacts/evaluation/statistical-report.json:power/
  rct_requirement/n_per_arm}}` becomes 176. A pointer that does not resolve
  fails the build.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PLACEHOLDER = re.compile(r"\{\{artifact:([^:}]+):([^}]*)\}\}")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(\s+\"[^\"]*\")?\)")


def resolve(path: str, pointer: str) -> str:
    """One value out of one artifact, formatted the way a document wants it."""
    target = ROOT / path
    if not target.exists():
        raise SystemExit(f"{path}: no such artifact (placeholder {pointer})")
    data = json.loads(target.read_text(encoding="utf-8"))
    for key in [part for part in pointer.split("/") if part]:
        if isinstance(data, list):
            data = data[int(key)]
        elif key in data:
            data = data[key]
        else:
            raise SystemExit(f"{path}: no key {key!r} on the way to {pointer!r}")
    if isinstance(data, float):
        return f"{data:.4g}"
    if isinstance(data, bool):
        return "yes" if data else "no"
    if isinstance(data, (list, dict)):
        raise SystemExit(f"{path}:{pointer} resolves to a {type(data).__name__}, "
                         "not a value a sentence can contain")
    return str(data)


def substitute(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        count += 1
        return resolve(match.group(1), match.group(2))

    return PLACEHOLDER.sub(replace, text), count


def check_images(text: str, source: Path) -> list[str]:
    missing, found = [], []
    for match in IMAGE.finditer(text):
        target = (source.parent / match.group(2)).resolve()
        if not target.exists():
            target = (ROOT / match.group(2)).resolve()
        (found if target.exists() else missing).append(match.group(2))
    if missing:
        raise SystemExit(f"{source.name}: {len(missing)} figure(s) not on disk: "
                         + ", ".join(sorted(set(missing))[:6]))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", type=Path)
    parser.add_argument("--css", type=Path, default=ROOT / "docs" / "reports" / "report.css")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--html", type=Path, default=None,
                        help="also keep the intermediate HTML (debugging)")
    arguments = parser.parse_args()

    source = arguments.markdown if arguments.markdown.is_absolute() else ROOT / arguments.markdown
    output = arguments.out or source.with_suffix(".pdf")
    output = output if output.is_absolute() else ROOT / output
    text = source.read_text(encoding="utf-8")
    text, substitutions = substitute(text)
    figures = check_images(text, source)

    import markdown as markdown_module

    body = markdown_module.markdown(
        text, extensions=["tables", "fenced_code", "toc", "attr_list", "footnotes",
                          "sane_lists", "md_in_html"])
    title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")),
                 source.stem)
    css = (arguments.css if arguments.css.is_absolute() else ROOT / arguments.css)
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{css.read_text(encoding='utf-8')}</style></head><body>{body}</body></html>")
    if arguments.html:
        arguments.html.write_text(html, encoding="utf-8")

    try:
        from weasyprint import HTML
    except ImportError:                                    # pragma: no cover
        raise SystemExit("weasyprint is not installed; run this inside the ml-service "
                         "container (`make report`) or `pip install weasyprint`")

    output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(source.parent)).write_pdf(str(output))
    size = output.stat().st_size / 1e6
    shown = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    print(f"wrote: {shown} ({size:.1f} MB, {len(figures)} figures, "
          f"{substitutions} numbers substituted from artifacts/)")


if __name__ == "__main__":
    sys.exit(main())
