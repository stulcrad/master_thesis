"""
Split a multi-subset dataset into N balanced groups, one per parallel process.

Printed one group per line, ready to append to a command line:

    --subset en_ewt de_pud hr_set          # uner
    --wmt-subset en-is-news en-ja-news     # wmt

The packing happens exactly ONCE, here. When an arm is too slow to finish in one job, the submitter splits it into
CHUNKS jobs and passes --chunk i/N; the subsets are then packed into (N * shards) groups in a single global pass and
this job is handed groups [i*shards, (i+1)*shards).

Preview a split without submitting anything:

    python -m utils.bin_packing uner 6 --max-examples 250
    python -m utils.bin_packing uner 6 --chunk 0/3 --max-examples 250
    python -m utils.bin_packing wmt 4
"""
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "semin"

# dataset -> (directory, filename prefix, the eval script's subset flag). The prefix is both the glob stem and what is
# stripped to recover the subset name, so uner_en_ewt.json -> "en_ewt" and wmt-en-is-news.json -> "en-is-news".
DATASETS = {
    "uner": (DATA_DIR / "universal_ner", "uner_", "--subset"),
    "wmt": (DATA_DIR / "wmt", "wmt-", "--wmt-subset"),
}


def subset_sizes(dataset, cap=None, only=None):
    """{subset: effective example count}. `cap` mirrors --max-examples, which is
    applied PER SUBSET, so packing must use the capped size or it balances against work that will never run."""
    directory, prefix, _ = DATASETS[dataset]
    sizes = {}
    for path in sorted(directory.glob(f"{prefix}*.json")):
        name = path.stem[len(prefix):]
        if only and name not in only:
            continue
        n = len(json.loads(path.read_text(encoding="utf-8")))
        sizes[name] = min(n, cap) if cap else n
    return sizes


def pack(sizes, n_groups):
    """Greedy longest-processing-time bin packing: largest subset first, always
    into the currently lightest group. Returns only NON-EMPTY groups, so asking for more groups than subsets yields
    fewer processes rather than idle ones that would each still load a full model."""
    groups = [[] for _ in range(n_groups)]
    loads = [0] * n_groups
    for name in sorted(sizes, key=lambda s: -sizes[s]):
        i = loads.index(min(loads))
        groups[i].append(name)
        loads[i] += sizes[name]
    return [g for g in groups if g]


def _parse(argv, subset_flag):
    """Pull --max-examples / the dataset's subset flag / --chunk out of the arguments,
    so the caller can pass "$@" straight through and the two can never disagree."""
    cap, only, chunk = None, None, (0, 1)
    if "--chunk" in argv:
        i = argv.index("--chunk")
        c, n = argv[i + 1].split("/")
        chunk = (int(c), int(n))
        if not (0 <= chunk[0] < chunk[1]):
            raise SystemExit(f"--chunk must be i/N with 0 <= i < N, got {argv[i + 1]!r}")
    if "--max-examples" in argv:
        i = argv.index("--max-examples")
        if i + 1 < len(argv):
            cap = int(argv[i + 1])
    if subset_flag in argv:
        i = argv.index(subset_flag)
        vals = []
        for a in argv[i + 1:]:
            if a.startswith("--"):
                break
            vals.append(a)
        if vals and vals != ["all"]:
            only = set(vals)
    return cap, only, chunk


def main(argv):
    if len(argv) < 2:
        raise SystemExit(f"usage: python -m utils.bin_packing <{'|'.join(DATASETS)}> <n_groups> [eval args...]")
    dataset = argv[0]
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset {dataset!r}, expected one of {', '.join(DATASETS)}")
    directory, _, subset_flag = DATASETS[dataset]
    shards = int(argv[1])
    # Get the maximum number of examples to use per subset, the optional subset list, and the optional --chunk i/N spec.
    cap, only, (chunk_i, chunk_n) = _parse(argv[2:], subset_flag)
    sizes = subset_sizes(dataset, cap, only)
    if not sizes:
        raise SystemExit(f"no {dataset} subsets found under {directory}")

    # ONE global packing over every process in the whole arm, then hand back only
    # this job's window of it.
    groups = pack(sizes, shards * chunk_n)
    mine = groups[chunk_i * shards:(chunk_i + 1) * shards]
    for group in mine:
        print(f"{subset_flag} " + " ".join(sorted(group)))


if __name__ == "__main__":
    main(sys.argv[1:])
