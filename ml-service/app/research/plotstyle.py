"""One figure style for every phase — palette, resolution, dual-format output.

Import it for the rcParams and call :func:`save` instead of ``fig.savefig``::

    from app.research import plotstyle

    SOURCES = ["artifacts/evaluation/policy-results-v0.json"]
    plotstyle.save(fig, FIG_DIR, "f08-01_policy-comparison.png")

Every save writes a 300 DPI PNG **and** a vector PDF beside it, and appends one
row to ``artifacts/figures/manifest.jsonl``: the figure, its caption, the script
that drew it and the files that script reads. ``app.research.figure_index``
renders ``artifacts/figures/INDEX.md`` from that manifest, so a figure's
provenance line is recorded by the code that made the picture rather than typed
into a document by hand and left to rot.

The palette is Okabe–Ito: eight hues that stay distinguishable under
deuteranopia, protanopia and tritanopia, and in greyscale by lightness. The
project previously used seaborn's ``deep``, whose blue/green/red trio is the
classic confusion set.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
FIGURE_ROOT = ROOT / "artifacts" / "figures"
MANIFEST = FIGURE_ROOT / "manifest.jsonl"

#: Okabe & Ito (2008), "Color Universal Design". Ordered so that the first
#: three — the ones a two- or three-series chart reaches for — are also the
#: most separated in lightness.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
           "#56B4E9", "#D55E00", "#F0E442", "#000000"]

BLUE, ORANGE, GREEN, PURPLE, SKY, VERMILLION, YELLOW, BLACK = PALETTE
GREY = "#666666"
LIGHT_GREY = "#cccccc"

#: What the old seaborn hex codes map to, kept so a reviewer can check that the
#: recolouring was a substitution and not a redraw.
LEGACY_MAP = {"#4c72b0": BLUE, "#55a868": GREEN, "#c44e52": VERMILLION,
              "#8172b3": PURPLE, "#dd8452": ORANGE, "#937860": SKY}

#: Figures are read on screen and printed. 300 DPI is the print floor; the PDF
#: beside every PNG is the one to embed in the paper.
DPI = 300


def apply() -> None:
    """Set the rcParams every figure in this project is drawn with."""
    plt.rcParams.update({
        "figure.dpi": 110,            # on-screen preview; savefig.dpi is what ships
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.autolayout": False,
        "pdf.fonttype": 42,           # embed TrueType, not Type 3: reviewers' viewers
    })


apply()


def caption_of(fig: plt.Figure) -> str:
    """The figure's own words: its suptitle, else its first axes title."""
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text().strip():
        return suptitle.get_text().strip().replace("\n", " ")
    for axis in fig.get_axes():
        if axis.get_title().strip():
            return axis.get_title().strip().replace("\n", " ")
    return ""


def _caller_module() -> tuple[str, list[str]]:
    """(script name, its declared SOURCES) for the module that called save()."""
    for frame in inspect.stack()[1:]:
        namespace = frame.frame.f_globals
        name = namespace.get("__name__", "")
        if name != __name__:
            script = namespace.get("__file__", name)
            try:
                script = str(Path(script).resolve().relative_to(ROOT))
            except (ValueError, OSError):
                script = name
            return script, list(namespace.get("SOURCES", []))
    return "unknown", []


def _record(entry: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def save(fig: plt.Figure, fig_dir: Path, name: str, caption: str | None = None,
         sources: list[str] | None = None, close: bool = True) -> Path:
    """Write ``name`` as a 300 DPI PNG and a PDF, and record its provenance.

    Returns the PNG path. ``fig_dir`` is created if it does not exist.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / name
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    script, declared = _caller_module()
    _record({
        "figure": png.name,
        "directory": str(fig_dir.relative_to(FIGURE_ROOT)) if fig_dir.is_relative_to(FIGURE_ROOT)
        else str(fig_dir),
        "caption": caption if caption is not None else caption_of(fig),
        "script": script,
        "sources": sources if sources is not None else declared,
    })
    if close:
        plt.close(fig)
    return png


def simulated_note(axis: plt.Axes, text: str = "SIMULATED") -> None:
    """Guardrail 6: a simulated result says so on its own face."""
    axis.text(0.99, 0.02, text, transform=axis.transAxes, ha="right", va="bottom",
              fontsize=7, color=PURPLE, alpha=0.9, fontweight="bold")
