"""Μέρος Α: nested subsets του degraded training set (dose-response μεγέθους).

Παίρνει τα πρώτα 25/50/75% των records — nested by construction
(25% ⊂ 50% ⊂ 75% ⊂ 100%), αφού το train_degraded.jsonl έχει σταθερή σειρά.
Το 100% είναι το ίδιο το train_degraded.jsonl (δεν ξαναγράφεται).
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="train_degraded.jsonl")
    parser.add_argument("--outdir", default="data/A_size")
    parser.add_argument("--fractions", default="0.25,0.5,0.75")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    total = len(lines)
    os.makedirs(args.outdir, exist_ok=True)

    for frac in [float(x) for x in args.fractions.split(",")]:
        k = int(round(total * frac))
        pct = int(round(frac * 100))
        out_path = os.path.join(args.outdir, f"train_deg_{pct}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fout:
            fout.writelines(lines[:k])
        print(f"{out_path}: {k}/{total} records ({pct}%)")


if __name__ == "__main__":
    main()
