"""
Measure every table in paper/ against the width it is actually allowed, and list what the paper contains.

Why this is not just "read the build log"
-----------------------------------------
An over-wide table is an Overfull \\hbox buried in a 30k-line main.log -- and that log is ISO-8859 encoded, so plain
grep treats it as binary and prints NOTHING unless you remember `grep -a`. The warning is also unitless about what
would fix it: "37pt too wide" does not say whether that is two characters or a whole column. This prints the natural
width, the limit, and the slack, per table, so the fix is a decision rather than a guessing game.

How it measures
---------------
The preamble is taken VERBATIM from paper/main.tex (everything up to \\begin{document}), so the measurement uses the
same class options, packages, fonts and \\newcommand definitions as the real build -- no second copy to drift. Each
table's float body (its \\centering / size / \\tabcolsep settings plus the tabular, minus caption and label) is put in
a \\savebox and its \\wd reported. All tables are measured in ONE pdflatex pass.

The limit depends on the float:
    table*   spans both columns -> must fit \\textwidth   (455.2pt in the ACL style)
    table    sits in one column -> must fit \\columnwidth (219.1pt)

This is deliberately the CONSERVATIVE number
--------------------------------------------
What is reported is the tabular's NATURAL width. main.tex loads microtype, which lets TeX shrink glyphs to claw back
roughly 10-15pt on a full-width table, so LaTeX's own "Overfull \\hbox (Npt too wide)" is normally smaller than the
deficit shown here. Taking microtype out of the preamble to measure would be worse, not better: it would understate
every width by the same amount. Treat a small negative slack as "fix it", not "the build will fail" -- the build log
stays the authority on whether a box is actually overfull. On the six tables as of this writing the two agree on
every pass/fail verdict.

Usage
-----
    python scripts/analysis/check_table_widths.py            # measure and report
    python scripts/analysis/check_table_widths.py --list     # inventory only, no LaTeX run
    python scripts/analysis/check_table_widths.py --keep     # leave the generated .tex/.log for inspection

Exits non-zero if any table overflows, so it can gate the final compliance sweep.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper"
MAIN = PAPER / "main.tex"
TABLES = PAPER / "tables"
BUILD = PAPER / "build.sh"

# TeX Live lives on NFS here, so a cold run is slow; one pass for all tables rather than one pass each.
TIMEOUT = 600


def module_load_line():
    """
    The `module load texlive/...` line, read out of build.sh rather than pinned again here.

    build.sh is what produces the submission PDF, so it owns the TeX Live version; duplicating the string would let
    this script measure against a different one than the paper is built with.
    """
    for line in BUILD.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("module load"):
            return line.strip()
    raise SystemExit(f"no 'module load' line in {BUILD} -- has the build script changed?")


def preamble():
    """main.tex up to (not including) \\begin{document}. Verbatim, so packages and \\newcommands cannot drift."""
    text = MAIN.read_text(encoding="utf-8")
    cut = text.find(r"\begin{document}")
    if cut < 0:
        raise SystemExit(rf"no \begin{{document}} in {MAIN}")
    return text[:cut]


def _strip_braced(text, command):
    """Remove every `\\command{...}` from `text`, counting braces so nested groups do not end the match early."""
    while True:
        start = text.find(command + "{")
        if start < 0:
            return text
        i = start + len(command) + 1
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        text = text[:start] + text[i:]


def extract(path):
    """
    -> (starred, body, label) for one table file, or None if it holds no float.

    `body` is the float's contents with the caption and label removed: the size and \\tabcolsep settings stay, because
    they are exactly what determines the width. `starred` says which limit applies.
    """
    text = path.read_text(encoding="utf-8")
    float_start = re.search(r"\\begin\{(table\*?)\}", text)
    tabular = re.search(r"\\begin\{tabular\*?\}", text)
    if not float_start or not tabular:
        return None
    label = re.search(r"\\label\{([^}]*)\}", text)
    end = text.rfind(r"\end{" + float_start.group(1) + "}")
    body = text[float_start.end():end if end > 0 else len(text)]
    body = _strip_braced(body, r"\caption")
    body = _strip_braced(body, r"\label")
    return float_start.group(1).endswith("*"), body.strip(), (label.group(1) if label else "?")


def measure(tables, keep):
    """
    -> {name: width_in_pt}, plus the two page limits, from a single pdflatex pass.

    Runs with cwd=paper/ so acl.sty and friends resolve the same way they do for the real build, and writes every
    auxiliary file into a temp directory so nothing lands next to the paper.
    """
    lines = [preamble(), r"\begin{document}", r"\newsavebox{\measurebox}",
             r"\typeout{MEASURE textwidth \the\textwidth}",
             r"\typeout{MEASURE columnwidth \the\columnwidth}"]
    for name, (_, body, _) in tables.items():
        lines.append(r"\savebox{\measurebox}{" + body + "}")
        lines.append(rf"\typeout{{MEASURE {name} \the\wd\measurebox}}")
    lines.append(r"\end{document}")

    workdir = Path(tempfile.mkdtemp(prefix="tablewidth-"))
    source = workdir / "measure.tex"
    source.write_text("\n".join(lines), encoding="utf-8")

    command = f"pdflatex -interaction=nonstopmode -draftmode -output-directory={workdir} {source}"
    if not shutil.which("pdflatex"):
        # Same two-step lmod dance as build.sh: `module` is a shell function, and lmod.sh reads SLURM_NODELIST
        # unguarded so it cannot run under `set -u`.
        command = f"source /etc/profile.d/lmod.sh && {module_load_line()} && {command}"
    result = subprocess.run(["bash", "-c", command], cwd=PAPER, capture_output=True, text=True, timeout=TIMEOUT)

    widths = {}
    for match in re.finditer(r"MEASURE (\S+) ([\d.]+)pt", result.stdout):
        widths[match.group(1)] = float(match.group(2))
    if "textwidth" not in widths:
        tail = "\n".join(result.stdout.splitlines()[-25:])
        raise SystemExit(f"pdflatex produced no measurements. Last output:\n{tail}")
    if keep:
        print(f"  (kept {workdir})\n")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return widths


def main():
    parser = argparse.ArgumentParser("Measure the paper's tables against the width they are allowed.")
    parser.add_argument("--list", action="store_true", help="Inventory the tables without running LaTeX.")
    parser.add_argument("--keep", action="store_true", help="Keep the generated .tex/.log for inspection.")
    args = parser.parse_args()

    tables = {}
    for path in sorted(TABLES.glob("*.tex")):
        found = extract(path)
        if found is None:
            print(f"  [skip] {path.name}: no table float found")
            continue
        tables[path.stem] = found
    if not tables:
        raise SystemExit(f"no tables found under {TABLES}")

    if args.list:
        print(f"{len(tables)} tables in {TABLES.relative_to(REPO)}\n")
        print(f"  {'file':32}{'label':20}{'float':8}rows")
        for name, (starred, body, label) in tables.items():
            rows = body.count(r"\\")
            print(f"  {name + '.tex':32}{label:20}{'table*' if starred else 'table':8}{rows}")
        return

    widths = measure(tables, args.keep)
    textwidth, columnwidth = widths["textwidth"], widths["columnwidth"]
    print(f"Page limits: \\textwidth {textwidth:.1f}pt (table*), \\columnwidth {columnwidth:.1f}pt (table)\n")
    print(f"  {'file':32}{'label':20}{'float':8}{'width':>9}{'limit':>9}{'slack':>9}")

    overflowing = []
    for name, (starred, _, label) in tables.items():
        width = widths.get(name)
        if width is None:
            print(f"  {name + '.tex':32}{label:20}{'':8}{'not measured':>27}")
            continue
        limit = textwidth if starred else columnwidth
        slack = limit - width
        if slack < 0:
            overflowing.append((name, -slack))
        flag = "" if slack >= 0 else "   <-- OVERFULL"
        print(f"  {name + '.tex':32}{label:20}{'table*' if starred else 'table':8}"
              f"{width:8.1f}p{limit:8.1f}p{slack:8.1f}p{flag}")

    print()
    if overflowing:
        for name, over in overflowing:
            print(f"  {name}.tex is {over:.1f}pt over its natural width. Shorten the widest cell of a text column "
                  f"or drop a column; reducing the font size buys far less than it looks like it should, because "
                  f"the width sits in the text columns rather than the glyphs.")
        raise SystemExit(f"{len(overflowing)} table(s) overflow.")
    print("  every table fits.")


if __name__ == "__main__":
    main()
