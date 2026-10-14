"""Inventory of every experiment arm: what is on disk, what is in flight, what is missing.

Three independent sources, none of them guessed:

1. RESULTS   -- Experiment_results_publication/<DS>/<Approach>/Csv/*.csv. One CSV per
                (arm, shard); its own columns carry model/bs/reasoning/effort/seeds/ n_iters/max_examples, and
                the `dataset` column carries the subset (conll2003_all, uner_en_ewt, wmt_en-cs-news, ...). The
                shard index comes from the filename tag `_shard<i>of<N>_`.
2. IN FLIGHT -- `sacct --format=SubmitLine`, parsed for --array=<idx>, SHARDS=<n> and
                the arm flags. This is the only way to tell a queued arm from a missing one; it is live
                cluster state and goes stale the moment anything runs.
3. EXPECTED  -- the ACTIVE (uncommented) `submit ...` lines in slurm/scripts/*/submit_*.sh.
                Arg order is read from each script's own `local part=$1 ...` line rather than assumed, because
                the header comments in two scripts are stale.

Usage:
    python scripts/analysis/audit_experiments.py            # full report python
    scripts/analysis/audit_experiments.py --since 2026-09-01 python scripts/analysis/audit_experiments.py
    --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "Experiment_results_publication"
sys.path.insert(0, str(REPO / "src"))

# idx -> model id, straight from the registry the batch scripts themselves call.
from utils.model_registry import MODEL_IDS  # noqa: E402

# submit-script directory -> (results dataset dirs it writes to)
SCRIPT_TO_DATASETS = {
    "NER": ["CoNLL", "UNER"],
    "ToxicSpans": ["ToxicSpans"],
    "LegalQA": ["LegalQAEval"],
    "GEC": ["GEC"],
    "ESA-MT": ["ESA-MT"],
}
SEEDS_TARGET = 3

# Two different ways work gets split, so two different completeness tests: index-shard
#   (CoNLL/ToxicSpans/LegalQAEval/GEC) -- `--shard i/N` over example indices; the filename carries
#                `_shard<i>of<N>_`. NOTE the tag was 0-based before commit 6dbc4d3 and 1-based after, so files
#                on disk mix both and the test must be base-agnostic. subset-split (UNER/ESA-MT) -- `--subset
#                a b c` / `--wmt-subset a b c`; the filename carries `_<n>subsets_<first>_to_<last>_` and
#   completeness means every treebank/file appears, not that shard ids are contiguous.
SUBSET_SPLIT = {
    "UNER": 18,    # len(uner_subsets())
    "ESA-MT": 24,  # len(wmt_subsets())
}


# --------------------------------------------------------------------------- #
# 1. RESULTS --------------------------------------------------------------------------- #
def arm_key(row, bs_in_name=None):
    """Identity of an arm, ignoring which shard/subset it is."""
    eff = str(row.get("reasoning_effort", "n|a"))
    reasoning = bool(row.get("reasoning_enabled", False))
    arm = eff if eff not in ("n|a", "nan") else ("ON" if reasoning else "OFF")
    return (
        str(row["model"]).split("/")[-1],
        int(row.get("batch_size", 1)),
        arm,
    )


def scan_results():
    """-> {(dataset, approach, model, bs, arm): {...coverage...}}"""
    out = {}
    for ds_dir in sorted(RESULTS.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name == "Archived_results":
            continue
        for ap_dir in sorted(ds_dir.iterdir()):
            csvs = sorted((ap_dir / "Csv").glob("*.csv")) if ap_dir.is_dir() else []
            if not csvs:
                continue
            for f in csvs:
                m = re.search(r"_shard(\d+)of(\d+)_", f.name)
                shard_i, shard_n = (int(m.group(1)), int(m.group(2))) if m else (1, 1)
                try:
                    df = pd.read_csv(f)
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] unreadable {f.name}: {e}", file=sys.stderr)
                    continue
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    k = (ds_dir.name, ap_dir.name) + arm_key(row)
                    e = out.setdefault(
                        k,
                        dict(
                            shard_sets=defaultdict(set),
                            subsets=set(),
                            seeds=set(),
                            modes=set(),
                            max_examples=set(),
                            files=set(),
                        ),
                    )
                    e["shard_sets"][shard_n].add(shard_i)
                    e["subsets"].add(str(row.get("dataset", "?")))
                    e["seeds"].add(int(row.get("n_iters", 0)))
                    e["modes"].add(str(row.get("eval_mode", "?")))
                    mx = row.get("max_examples")
                    e["max_examples"].add(None if pd.isna(mx) else int(mx))
                    e["files"].add(f.name)
    return out


# --------------------------------------------------------------------------- #
# 2. IN FLIGHT --------------------------------------------------------------------------- #
def scan_sacct(since):
    fmt = "JobID,JobName%26,State%14,Elapsed,Submit,SubmitLine%600"
    try:
        raw = subprocess.run(
            ["sacct", "-X", "-S", since, "--parsable2", "--noheader", f"--format={fmt}"],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except Exception as e:  # noqa: BLE001
        print(f"[warn] sacct unavailable: {e}", file=sys.stderr)
        return []
    jobs = []
    for line in raw.splitlines():
        p = line.split("|")
        if len(p) < 6:
            continue
        jobid, name, state, elapsed, submit, sline = p[0], p[1], p[2], p[3], p[4], p[5]
        if "submit_" in name or not sline.strip():
            continue  # the submitter wrapper job itself, not an experiment
        script = re.search(r"slurm/scripts/([^/]+)/(\S+)", sline)
        if not script:
            continue
        idx = re.search(r"--array=(\d+)", sline)
        shards = re.search(r"SHARDS=(\d+)", sline)
        gpus = re.search(r"--gres=gpu:(\d+)", sline)
        bs = re.search(r"--batch_size (\d+)", sline)
        eff = re.search(r"--reasoning-effort (\w+)", sline)
        thinking = "--no-enable-thinking" in sline
        jobs.append(dict(
            jobid=jobid, name=name, state=state.split()[0], elapsed=elapsed, submit=submit,
            script_dir=script.group(1), script=script.group(2),
            idx=int(idx.group(1)) if idx else None,
            model=MODEL_IDS[int(idx.group(1))].split("/")[-1] if idx and int(idx.group(1)) < len(MODEL_IDS) else "?",
            shards=int(shards.group(1)) if shards else 1,
            gpus=int(gpus.group(1)) if gpus else None,
            bs=int(bs.group(1)) if bs else 1,
            effort=eff.group(1) if eff else None,
            arm=(eff.group(1) if eff else ("OFF" if thinking else "ON")),
            subset=" ".join(re.findall(r"--(?:uner-)?subset ([\w\- ]+)", sline))[:60] or None,
            dataset=re.search(r"--dataset (\w+)", sline).group(1) if "--dataset " in sline else None,
            # An arm is routinely SPLIT over several jobs, and the split is carried by one of these three,
            # never by the arm flags. Two jobs that differ only here are complementary halves, NOT duplicates
            # -- leaving them out of the identity makes the duplicate check fire on every split arm.
            piece=(
                (re.search(r"--shard (\d+/\d+)", sline) or [None, None])[1]
                if "--shard " in sline else None,
                (re.search(r"WMT_CHUNK=(\d+/\d+)", sline) or [None, None])[1]
                if "WMT_CHUNK=" in sline else None,
                (re.search(r"UNER_CHUNK=(\d+/\d+)", sline) or [None, None])[1]
                if "UNER_CHUNK=" in sline else None,
            ),
        ))
    return jobs


# --------------------------------------------------------------------------- #
# 3. EXPECTED (active submit lines)
# --------------------------------------------------------------------------- #
def parse_submit_scripts():
    expected = []
    for sdir, _ in SCRIPT_TO_DATASETS.items():
        for script in sorted((REPO / "slurm" / "scripts" / sdir).glob("submit_*.sh")):
            text = script.read_text().splitlines()
            sig = next((l for l in text if "local part=" in l), "")
            names = re.findall(r"(\w+)=\$\d", sig)  # e.g. part gpus idx mnt wtime shards
            bs = 1
            for line in text:
                s = line.strip()
                if re.match(r"^BS=\d+", s):
                    bs = int(re.match(r"^BS=(\d+)", s).group(1))
                if s.startswith("#") or not s.startswith("submit "):
                    continue
                toks = s.split()[1:]
                pos = dict(zip(names, toks))
                extra = toks[len(names):]
                eff = None
                if "--reasoning-effort" in extra:
                    eff = extra[extra.index("--reasoning-effort") + 1]
                arm = eff or ("OFF" if "--no-enable-thinking" in extra else "ON")
                # An arm can be spread over several explicit lines, one per `--shard i/N` -- those are
                # distinct work units, not duplicates.
                shard = extra[extra.index("--shard") + 1] if "--shard" in extra else None
                idx = int(pos.get("idx", -1))
                expected.append(dict(
                    script_dir=sdir, script=script.name, bs=bs,
                    idx=idx,
                    model=MODEL_IDS[idx].split("/")[-1] if 0 <= idx < len(MODEL_IDS) else "?",
                    arm=arm, partition=pos.get("part"), gpus=pos.get("gpus"),
                    wtime=pos.get("wtime"), shards=pos.get("shards"), shard=shard,
                ))
    return expected


# --------------------------------------------------------------------------- #
def shard_status(shard_sets):
    """(text, ok) -- base-agnostic: a shard set is whole iff it is a contiguous run
    of length N starting at 0 or at 1 (the tag base changed in commit 6dbc4d3)."""
    parts, ok = [], True
    for n in sorted(shard_sets):
        got = sorted(shard_sets[n])
        whole = got in (list(range(n)), list(range(1, n + 1)))
        if not whole:
            base = 0 if (got and got[0] == 0) else 1
            missing = [i for i in range(base, base + n) if i not in shard_sets[n]]
            parts.append(f"{len(got)}/{n} missing{missing}")
            ok = False
        else:
            parts.append(f"{len(got)}/{n}")
    if len(shard_sets) > 1:  # same arm written under two different split factors
        ok = False
    return ", ".join(parts), ok


def subset_status(subsets, expected):
    got = len(subsets)
    return f"{got}/{expected} subsets", got == expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-08-25")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    res = scan_results()
    jobs = scan_sacct(args.since)
    exp = parse_submit_scripts()

    live = [j for j in jobs if j["state"] in ("RUNNING", "PENDING", "REQUEUED")]

    print("=" * 100)
    print("1. RESULTS ON DISK  (one row per arm; Csv files are the authority)")
    print("=" * 100)
    print(f"{'dataset':12}{'approach':17}{'model':16}{'bs':>4}{'arm':8}{'seeds':>7}  {'coverage':34} modes")
    complete, partial = [], []
    for k in sorted(res):
        ds, ap_, model, bs, arm = k
        e = res[k]
        seeds = sorted(e["seeds"])
        if ds in SUBSET_SPLIT:
            cov, ok_cov = subset_status(e["subsets"], SUBSET_SPLIT[ds])
        else:
            cov, ok_cov = shard_status(e["shard_sets"])
        ok = ok_cov and seeds == [SEEDS_TARGET]
        (complete if ok else partial).append(k)
        reasons = []
        if not ok_cov:
            reasons.append("coverage")
        if seeds != [SEEDS_TARGET]:
            reasons.append(f"seeds={seeds}")
        flag = "" if ok else "   <-- INCOMPLETE (" + ", ".join(reasons) + ")"
        print(f"{ds:12}{ap_:17}{model:16}{bs:>4}{arm:8}{str(seeds):>7}  "
              f"{cov:34} {','.join(sorted(e['modes']))}{flag}")
    print(f"\n  complete arms: {len(complete)}   incomplete arms: {len(partial)}")

    print()
    print("=" * 100)
    print(f"2. IN FLIGHT  (sacct since {args.since})")
    print("=" * 100)
    if not live:
        print("  (nothing running or pending)")
    for j in sorted(live, key=lambda x: (x["script_dir"], x["idx"] or -1)):
        print(f"  {j['jobid']:14}{j['state']:9}{j['script_dir']:10}{j['model']:16}"
              f"bs={j['bs']:<3}arm={j['arm']:8}SHARDS={j['shards']:<3}gpu={j['gpus']}  "
              f"{j['dataset'] or ''} {j['subset'] or ''}")

    # Two live jobs for one arm write to the SAME Csv/Predictions paths and will interleave/corrupt each
    # other. This has happened before, so check explicitly.
    dup = defaultdict(list)
    for j in live:
        dup[(j["script_dir"], j["model"], j["bs"], j["arm"], j["dataset"],
             j["shards"], j["piece"])].append(j["jobid"])
    dups = {k: v for k, v in dup.items() if len(v) > 1}
    print()
    if dups:
        print("  !! DUPLICATE LIVE JOBS -- same arm queued more than once (same output paths) !!")
        for k, v in sorted(dups.items()):
            print(f"     {k[0]:10}{k[1]:16}bs={k[2]:<3}arm={k[3]:8}{k[4] or chr(32)*10:10}SHARDS={k[5]:<3}piece={k[6]} -> {v}")
    else:
        print("  no duplicate live jobs")

    print()
    print("  --- terminal states since then, per arm (latest attempt wins) ---")
    agg = defaultdict(list)
    for j in jobs:
        if j["state"] in ("RUNNING", "PENDING", "REQUEUED"):
            continue
        agg[(j["script_dir"], j["model"], j["bs"], j["arm"])].append((j["submit"], j["state"], j["jobid"]))
    for k in sorted(agg):
        last = sorted(agg[k])[-1]
        states = {s for _, s, _ in agg[k]}
        print(f"  {k[0]:10}{k[1]:16}bs={k[2]:<3}arm={k[3]:8}last={last[1]:12}({last[2]})  all_seen={sorted(states)}")

    print()
    print("=" * 100)
    print("3. WHAT THE SUBMIT SCRIPTS WOULD LAUNCH NOW  (active, uncommented lines)")
    print("=" * 100)
    print(f"{'script':34}{'model':16}{'bs':>4}{'arm':8}{'part':12}{'gpu':>4}{'shards':>7}{'shard':>8} {'time'}")
    for e in exp:
        print(f"{e['script_dir'] + '/' + e['script']:34}{e['model']:16}{e['bs']:>4}{e['arm']:8}"
              f"{str(e['partition']):12}{str(e['gpus']):>4}{str(e['shards']):>7}"
              f"{(e.get('shard') or '-'):>8} {e['wtime']}")
    print(f"\n  active submit lines: {len(exp)}")

    if args.json:
        ser = {
            "results": [
                dict(dataset=k[0], approach=k[1], model=k[2], bs=k[3], arm=k[4],
                     seeds=sorted(res[k]["seeds"]),
                     shards={str(n): sorted(res[k]["shard_sets"][n]) for n in res[k]["shard_sets"]},
                     subsets=sorted(res[k]["subsets"]),
                     n_subsets=len(res[k]["subsets"]), modes=sorted(res[k]["modes"]),
                     complete=k in complete)
                for k in sorted(res)
            ],
            "in_flight": live,
            "expected_active_submit_lines": exp,
        }
        Path(args.json).write_text(json.dumps(ser, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
