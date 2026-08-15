"""Download and rebuild the datasets used by Semin et al. (2026), arXiv:2601.16946,
"Strategies for Span Labeling with Large Language Models".

Their repo (https://github.com/semindan/span_labeling) gitignores `data/`, so the
JSON files the experiments read are not published. They ARE reconstructible: the
repo ships the exact parsers, and three of the four raw sources are public.

This script fetches the raw sources, then runs *their* parsers on them (imported
from a checkout of their repo, not reimplemented) so the output is byte-identical
to what they evaluated on.

Reproduced exactly (verified against the counts reported in the paper):
    universal_ner/  18 files, 7523 examples   <- paper: 7,523
    wmt/            24 files,  867 examples   <- paper:   867
    multigec/        1 file,   504 examples   <- paper:   504

Not automatic:
    multigec/       504 examples. The English Write & Improve 2024 corpus is
                    licence-gated; see MULTIGEC_INSTRUCTIONS below. Once the .m2
                    file is in place this script converts it.
    synthetic/      Their CPL task. Regenerated, not downloaded -- see
                    --with-synthetic. Their generator seeds `random.seed(432)`
                    at import, but the paper does not state the exact generation
                    arguments, so a regenerated set is *not* guaranteed to be the
                    same 1000 examples they used. Treat it as a new sample.

Usage:
    python scripts/data/prepare_semin_data.py                 # NER + WMT
    python scripts/data/prepare_semin_data.py --with-multigec # also convert a .m2 you placed
    python scripts/data/prepare_semin_data.py --with-synthetic
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "semin"
RAW_ROOT = DATA_ROOT / "raw"

# The semin repo's and WMT repo
SPAN_LABELING_REPO = "https://github.com/semindan/span_labeling.git"
SPAN_ANNOTATION_REPO = "https://github.com/llm-span-annotators/span-annotation.git"

# Pinned so a rerun months from now rebuilds byte-identical data.
SPAN_LABELING_SHA = "11e86b85efcc228953764afce1cb7700dd0aafbb"    # 2026-05-22
SPAN_ANNOTATION_SHA = "695e033aa0fbb8a7b0031a0dbf8bdc827056b9ad"  # 2026-01-30

# raw.githubusercontent.com 404s on urllib's default User-Agent.
UA = {"User-Agent": "Mozilla/5.0"}

# Original UniversalNER profile, languages are stored as their own repos
UNER_PREFIX = "https://raw.githubusercontent.com/UniversalNER/"

# The 18 UNER test sets in their NER_ALL group. The repo/filename mapping comes
# from the loader script of the `universalner/universal_ner` HF dataset; that
# dataset is script-based and no longer loadable under datasets>=3, so we fetch
# the .iob2 files straight from GitHub instead.
#
# Each SHA is the newest commit touching that test file at or before 2023-11-14 --
# the timestamp embedded in the UNER archive name in their convert_data.py
# (`uner-20231114-092426`), i.e. the annotation state they evaluated on.
#
# Auditing all 18 histories, exactly ONE test file has changed since that date:
#   en_pud, commit aac717c7c8 (2025-10-21), a single token relabelled
#   `Germany` B-PER -> B-LOC. Byte sizes are otherwise identical.
# Pinning below reproduces their state; --use-latest gives you current master,
# which differs only by that one span (1 of 14,162 gold spans, 0.007%).
UNER_TEST_FILES = {
    "ceb_gja": ("UNER_Cebuano-GJA", "master", "c56cb85c21", "ceb_gja-ud-test.iob2"),
    "zh_gsd": ("UNER_Chinese-GSD", "master", "9696baf082", "zh_gsd-ud-test.iob2"),
    "zh_gsdsimp": ("UNER_Chinese-GSDSIMP", "master", "1fefc9dd59", "zh_gsdsimp-ud-test.iob2"),
    "zh_pud": ("UNER_Chinese-PUD", "master", "dae369246f", "zh_pud-ud-test.iob2"),
    "hr_set": ("UNER_Croatian-SET", "main", "d49832adc3", "hr_set-ud-test.iob2"),
    "da_ddt": ("UNER_Danish-DDT", "main", "b416409c35", "da_ddt-ud-test.iob2"),
    "en_ewt": ("UNER_English-EWT", "master", "89887002ed", "en_ewt-ud-test.iob2"),
    "en_pud": ("UNER_English-PUD", "master", "85cdc2938e", "en_pud-ud-test.iob2"),
    "de_pud": ("UNER_German-PUD", "master", "f2d508ddfa", "de_pud-ud-test.iob2"),
    "pt_bosque": ("UNER_Portuguese-Bosque", "master", "ccc4f904ec", "pt_bosque-ud-test.iob2"),
    "pt_pud": ("UNER_Portuguese-PUD", "master", "03a0137251", "pt_pud-ud-test.iob2"),
    "ru_pud": ("UNER_Russian-PUD", "master", "e1186ba65b", "ru_pud-ud-test.iob2"),
    "sr_set": ("UNER_Serbian-SET", "main", "da40b44a25", "sr_set-ud-test.iob2"),
    "sk_snk": ("UNER_Slovak-SNK", "master", "48e9657386", "sk_snk-ud-test.iob2"),
    "sv_pud": ("UNER_Swedish-PUD", "master", "984ddb8e1b", "sv_pud-ud-test.iob2"),
    "sv_talbanken": ("UNER_Swedish-Talbanken", "master", "8cdc07ee55", "sv_talbanken-ud-test.iob2"),
    "tl_trg": ("UNER_Tagalog-TRG", "master", "d32640dfc9", "tl_trg-ud-test.iob2"),
    "tl_ugnayan": ("UNER_Tagalog-Ugnayan", "master", "9d46d20fd4", "tl_ugnayan-ud-test.iob2"),
}

# The WMT24 ESA (error span annotation) task has three domains
WMT_DOMAINS = ["news", "literary", "social"]

# Downloaded from the Cambridge Write & Improve 2024 corpus, which is licence-gated.
MULTIGEC_M2_NAME = "en-writeandimprove2024-ref1-dev.m2"

MULTIGEC_INSTRUCTIONS = f"""
MultiGEC (English Write & Improve 2024) is licence-gated and cannot be fetched
automatically. To add it:

  1. Go to https://researchdatasets.cambridge.org/datasets/write-and-improve-corpus-2024
     and accept the licence (free, but requires an account).
     The MultiGEC-2025 shared task page mirrors the same data and lists the other
     languages: https://spraakbanken.github.io/multigec-2025/
  2. From the download, take the file
        multigec-2025-files/local_eval/ref/{MULTIGEC_M2_NAME}
  3. Copy it to
        {{dest}}
  4. Re-run this script with --with-multigec

Expected result: 504 examples (the count reported in the paper).
""".strip()


def log(msg):
    """Print a message to stdout, flushing immediately so it appears in logs."""
    print(msg, flush=True)


def fetch(url: str, dest: Path) -> None:
    """Download `url` to `dest` unless it is already there and non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        return # already there
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Make a request with a User-Agent header so GitHub doesn't 404 it.
    # The headers are needed because urllib's default User-Agent is blocked by GitHub.
    req = urllib.request.Request(url, headers=UA)
    # Download with a 60s timeout, in case the server is slow.
    # and write to the destination file.
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def clone(url: str, dest: Path, sha: str) -> Path:
    """Clone `url` to `dest` and check out the pinned `sha`."""
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"  cloning {url}")
        # Clone the repo, but suppress stdout/stderr to avoid spamming logs. The
        # `check=True` argument will raise an exception if the command fails.
        subprocess.run(
            ["git", "clone", url, str(dest)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    # Check out the pinned SHA, again suppressing output. This will fail if the SHA
    # is not in the repo, which is what we want: we want to ensure we are using
    # the exact version of the code they used.
    log(f"  checking out {sha} in {dest.name}")
    subprocess.run(
        ["git", "-C", str(dest), "checkout", "--quiet", sha],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Get the current HEAD commit hash to log it. This is useful for debugging and
    # for ensuring reproducibility. We capture the output and strip whitespace.
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    log(f"  {dest.name} @ {head[:10]}")
    return dest


def load_their_parsers(span_labeling_dir: Path):
    """Import convert_data.py from their checkout.

    Imported by file path rather than added to sys.path: the module has no
    imports from its own package, and this avoids their `span_labeling` package
    name colliding with anything of ours.
    """
    # Use importlib to load the module from the file path. This allows us to use their
    # exact parsers without having to modify sys.path or the module itself.
    path = span_labeling_dir / "span_labeling" / "datasets" / "convert_data.py"
    # The module name "semin_convert_data" is arbitrary; it just needs to be unique
    spec = importlib.util.spec_from_file_location("semin_convert_data", path)
    # Create a new module based on the spec and execute it to load the code.
    module = importlib.util.module_from_spec(spec)
    # Execute the module in its own namespace. This runs the code in convert_data.py
    spec.loader.exec_module(module)
    return module


def count_examples(directory: Path) -> int:
    """Count the total number of examples in all JSON files in `directory`."""
    return sum(len(json.loads(p.read_text())) for p in directory.glob("*.json"))


def build_ner(convert_data, use_latest: bool = False) -> int:
    """Download and convert the 18 UniversalNER test sets to JSON."""
    log("\n=== UniversalNER (18 test sets) ===")
    raw_dir = RAW_ROOT / "uner"
    out_dir = DATA_ROOT / "universal_ner"

    log(f"  revision: {'current master (NOT the state Semin used)' if use_latest else 'pinned to 2023-11-14 snapshot'}")
    # Download each of the 18 .iob2 test files from the UNER GitHub repos. The
    # `UNER_TEST_FILES` dictionary contains the repo, branch, SHA, and filename
    # for each test set. We use the branch if `use_latest` is True, otherwise
    # we use the pinned SHA to reproduce the exact state they evaluated on.
    for cfg, (repo, branch, sha, fname) in UNER_TEST_FILES.items():
        rev = branch if use_latest else sha
        fetch(f"{UNER_PREFIX}{repo}/{rev}/{fname}", raw_dir / f"{cfg}-ud-test.iob2")
    log(f"  downloaded {len(UNER_TEST_FILES)} .iob2 files -> {raw_dir}")

    # Convert each .iob2 file to JSON using their UniversalNERParser. The output
    # JSON files are written to the `out_dir`. The output filenames are prefixed
    # with "uner_" to distinguish them from other datasets.
    for cfg in UNER_TEST_FILES:
        convert_data.UniversalNERParser.parse(
            input_path=raw_dir / f"{cfg}-ud-test.iob2",
            output_path=out_dir / f"uner_{cfg}.json",
        )
    # Count the total number of examples across all JSON files in the output directory
    return count_examples(out_dir)


def build_wmt(convert_data, span_annotation_dir: Path) -> int:
    """Download and convert the WMT24 ESA (error span annotation) task to JSON."""
    log("\n=== WMT24 ESA (error span annotation) ===")
    out_dir = DATA_ROOT / "wmt"
    annotations = span_annotation_dir / "annotations" / "human" / "mt-eval" / "annotations.jsonl"

    # For each of the three WMT domains (news, literary, social), we look for the
    # corresponding input and output JSONL files in the span annotation directory. If
    # the output file does not exist for a given input, we log a warning and skip
    # that input. Otherwise, we use their WMTParser to convert the input and output
    # JSONL files, along with the annotations, into a single JSON file in the output
    # directory. The output filename includes the domain to distinguish between them.
    for domain in WMT_DOMAINS:
        inputs_dir = span_annotation_dir / "inputs" / "mt-eval" / f"wmt24-{domain}"
        outputs_dir = span_annotation_dir / "outputs" / "mt-eval" / f"wmt24-{domain}"
        for inputs_path in sorted(inputs_dir.glob("*.jsonl")):
            outputs_path = outputs_dir / inputs_path.name
            if not outputs_path.exists():
                log(f"  !! no outputs for {inputs_path.name} ({domain}), skipping")
                continue
            convert_data.WMTParser.parse(
                inputs_path=inputs_path,
                outputs_path=outputs_path,
                annotations_path=annotations,
                output_path=out_dir / f"wmt-{inputs_path.stem}-{domain}.json",
            )
    return count_examples(out_dir)


def build_multigec(convert_data) -> int | None:
    """Convert a manually-placed Write & Improve .m2 file to JSON."""
    log("\n=== MultiGEC (English Write & Improve 2024) ===")
    m2_path = RAW_ROOT / "multigec" / MULTIGEC_M2_NAME
    out_dir = DATA_ROOT / "multigec"

    # If the .m2 file does not exist, we log instructions for the user to download it
    # from the Cambridge Write & Improve 2024 corpus, which is licence-gated.
    if not m2_path.exists():
        m2_path.parent.mkdir(parents=True, exist_ok=True)
        log(MULTIGEC_INSTRUCTIONS.format(dest=m2_path))
        return None

    # Convert the .m2 file to JSON using their MultiGECParser. The output JSON file is
    # written to the `out_dir`. Since it is a single file, the parsing is straightforward.
    convert_data.MultiGECParser.parse(
        input_path=m2_path,
        output_path=out_dir / "multigec_en.json",
    )
    return count_examples(out_dir)


def build_synthetic(span_labeling_dir: Path) -> int | None:
    log("\n=== Synthetic CPL ===")
    log(
        "  Their generator lives at\n"
        f"    {span_labeling_dir / 'span_labeling' / 'datasets' / 'synthetic_generator'}\n"
        "  and seeds random.seed(432) at import, so it is deterministic -- but the\n"
        "  paper does not state the generation arguments used for the released\n"
        "  1000-example set, so regenerating gives a DIFFERENT sample.\n"
        "  Generate one with, e.g.:\n"
        "    python main.py --mode word --language en --num-examples 1000 \\\n"
        "        --output english_word_dataset.json\n"
        "  then run convert_data.SyntheticParser on it. Numbers produced this way\n"
        "  are NOT comparable to their published synthetic column."
    )
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-multigec", action="store_true", help="Convert a manually-placed Write & Improve .m2 file.")
    parser.add_argument("--with-synthetic", action="store_true", help="Print instructions for regenerating the synthetic CPL task.")
    parser.add_argument("--use-latest", action="store_true", help="Fetch current UNER master instead of the pinned 2023-11-14 snapshot Semin used.")
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    log("=== Fetching source repositories ===")
    span_labeling_dir = clone(SPAN_LABELING_REPO, RAW_ROOT / "span_labeling", SPAN_LABELING_SHA)
    span_annotation_dir = clone(SPAN_ANNOTATION_REPO, RAW_ROOT / "span-annotation", SPAN_ANNOTATION_SHA)
    convert_data = load_their_parsers(span_labeling_dir)

    summary = {}
    summary["universal_ner"] = (build_ner(convert_data, use_latest=args.use_latest), 7523)
    summary["wmt"] = (build_wmt(convert_data, span_annotation_dir), 867)

    if args.with_multigec:
        got = build_multigec(convert_data)
        if got is not None:
            summary["multigec"] = (got, 504)
    if args.with_synthetic:
        build_synthetic(span_labeling_dir)

    log("\n=== Summary ===")
    ok = True
    # Compare the number of examples we got with the expected counts from the paper.
    for name, (got, expected) in summary.items():
        mark = "OK " if got == expected else "!! " # If they match, mark as OK; otherwise, mark as a warning.
        if got != expected:
            ok = False
        log(f"  {mark}{name:16s} {got:>6,} examples (paper: {expected:,})")
    log(f"\nData written to {DATA_ROOT}")

    # Their published per-run results, for baseline comparison.
    results_csv = span_labeling_dir / "results" / "results.csv"
    if results_csv.exists():
        log(f"Their published results (16,875 runs): {results_csv}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
