"""
Compare two benchmark runs question-by-question.

Unlike compare_runs.py (hardcoded to the first 50 and to eval/results.csv, for
one specific prompt experiment), this takes both arms as arguments and reports
overall / per-difficulty / per-database accuracy, the flips in BOTH directions,
and McNemar's exact test on the discordant pairs.

The two-way flip count is the point. A net +2 can be 2 fixed and 0 broken, or
14 fixed and 12 broken -- the same headline, very different stories. And on a
small slice a net gain can easily be noise, so the p-value is reported rather
than left to the reader's optimism.

Usage:
    python -m scripts.compare_ablation BASELINE_CSV NEW_CSV [N] [--examples K]
"""

import csv
import sys
from math import comb


def load(path, n=None):
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    return rows[:n] if n else rows


def split_warning(baseline, new):
    """
    A message if the two runs are not known to be on the same split, else None.

    Runs written before splits existed have no split column; two of those are
    compared silently (that is the committed baseline and ablation pair). A
    mixed pair, or two different splits, answered different questions.
    """
    splits = [{r["split"] for r in rows if r.get("split")} for rows in (baseline, new)]
    if not splits[0] and not splits[1]:
        return None
    if splits[0] != splits[1]:
        shown = [sorted(s) or ["unknown (written before splits)"] for s in splits]
        return f"baseline split {shown[0]} vs new split {shown[1]}: these runs did not answer the same questions."
    return None


def is_correct(row):
    return str(row.get("correct")) == "True"


def mcnemar_exact_two_sided(fixed, broken):
    """
    Exact McNemar: under the null, each discordant pair is a fair coin, so the
    number fixed is Binomial(fixed + broken, 0.5). Two-sided p by summing the
    tail at least as extreme as observed. No scipy dependency.
    """
    n = fixed + broken
    if n == 0:
        return 1.0
    observed = min(fixed, broken)
    tail = sum(comb(n, k) for k in range(observed + 1))
    return min(1.0, 2 * tail / (2 ** n))


def group_accuracy(rows, key):
    groups = {}
    for r in rows:
        c, t = groups.setdefault(r[key], [0, 0])
        groups[r[key]] = [c + int(is_correct(r)), t + 1]
    return groups


def pct(c, t):
    return f"{c / t:.2%} ({c}/{t})" if t else "n/a"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        raise SystemExit(2)

    baseline_path, new_path = args[0], args[1]
    n = int(args[2]) if len(args) > 2 else None
    n_examples = 5
    if "--examples" in sys.argv:
        n_examples = int(sys.argv[sys.argv.index("--examples") + 1])

    new = load(new_path)
    n = n or len(new)
    baseline = load(baseline_path, n)
    new = new[:n]

    warning = split_warning(baseline, new)
    if warning:
        print(f"WARNING: {warning}")

    if len(baseline) != len(new):
        print(f"WARNING: comparing {len(baseline)} baseline rows against {len(new)} new rows")

    # Align on question_id rather than position, so a reordered or partial run
    # cannot silently compare two different questions.
    b_by_id = {r["question_id"]: r for r in baseline}
    paired = [(b_by_id[r["question_id"]], r) for r in new if r["question_id"] in b_by_id]
    unmatched = len(new) - len(paired)
    if unmatched:
        print(f"WARNING: {unmatched} rows in the new run had no baseline counterpart\n")

    b_rows = [b for b, _ in paired]
    n_rows = [x for _, x in paired]
    bc = sum(is_correct(r) for r in b_rows)
    nc = sum(is_correct(r) for r in n_rows)
    total = len(paired)

    print(f"Paired questions: {total}\n")
    print("OVERALL")
    print(f"  baseline : {pct(bc, total)}")
    print(f"  new      : {pct(nc, total)}")
    print(f"  delta    : {nc - bc:+d} questions ({(nc - bc) / total:+.2%})\n")

    for key, label in (("difficulty", "BY DIFFICULTY"), ("db_id", "BY DATABASE")):
        bg, ng = group_accuracy(b_rows, key), group_accuracy(n_rows, key)
        print(label)
        width = max(len(k) for k in bg) if bg else 10
        for name in sorted(bg, key=lambda k: -bg[k][1]):
            bcc, bt = bg[name]
            ncc, nt = ng.get(name, [0, 0])
            print(f"  {name:>{width}}: {pct(bcc, bt):>18} -> {pct(ncc, nt):>18}  ({ncc - bcc:+d})")
        print()

    fixed = [(b, x) for b, x in paired if is_correct(x) and not is_correct(b)]
    broken = [(b, x) for b, x in paired if is_correct(b) and not is_correct(x)]
    p = mcnemar_exact_two_sided(len(fixed), len(broken))

    print("FLIPS")
    print(f"  fixed  : {len(fixed)}")
    print(f"  broken : {len(broken)}")
    print(f"  net    : {len(fixed) - len(broken):+d}")
    print(f"  McNemar exact two-sided p = {p:.4f}"
          f" ({'significant' if p < 0.05 else 'not significant'} at 0.05)\n")

    for label, rows in (("FIXED", fixed), ("BROKEN", broken)):
        print(f"{label} ({len(rows)}) -- showing up to {n_examples}")
        for b, x in rows[:n_examples]:
            print(f"  [{x['db_id']}] qid={x['question_id']} ({x['difficulty']})")
            print(f"    Q        : {x['question'][:150]}")
            ev = (x.get("evidence") or "").strip()
            print(f"    evidence : {ev[:200] if ev else '(none)'}")
            print(f"    baseline : {(b.get('predicted_sql') or '')[:150]}")
            print(f"    new      : {(x.get('predicted_sql') or '')[:150]}")
            print(f"    gold     : {(x.get('gold_sql') or '')[:150]}")
            print()
        print()


if __name__ == "__main__":
    main()
