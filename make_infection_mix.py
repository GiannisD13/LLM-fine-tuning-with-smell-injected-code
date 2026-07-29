"""Μέρος Α2: μίξεις clean+degraded των ΙΔΙΩΝ αρχείων σε ποσοστά μόλυνσης.

Σταθερό μέγεθος (όλα τα paired αρχεία)· μεταβάλλεται μόνο πόσα αρχεία είναι
στη degraded εκδοχή τους. Shuffle με σταθερό seed και μετά first-N → οι
degraded του 25% ⊂ 50% ⊂ 75% (nested) και σκορπισμένες σε όλα τα repos.
Άκρα: 0% = train_clean.jsonl, 100% = train_degraded.jsonl (υπάρχουν ήδη).
"""
import argparse
import json
import os
import random


def load(path):
    content = {}
    order = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["repo"], rec["path"])
            content[key] = rec["content"]
            order.append(key)
    return content, order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", default="train_clean.jsonl")
    parser.add_argument("--degraded", default="train_degraded.jsonl")
    parser.add_argument("--outdir", default="data/A2_infect")
    parser.add_argument("--fractions", default="0.25,0.5,0.75")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clean, clean_order = load(args.clean)
    degraded, _ = load(args.degraded)
    keys = [k for k in clean_order if k in degraded]  # μόνο τα paired
    random.Random(args.seed).shuffle(keys)
    total = len(keys)
    os.makedirs(args.outdir, exist_ok=True)

    for frac in [float(x) for x in args.fractions.split(",")]:
        k_deg = int(round(total * frac))
        deg_set = set(keys[:k_deg])
        pct = int(round(frac * 100))
        out_path = os.path.join(args.outdir, f"train_inf_{pct}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fout:
            for key in keys:
                text = degraded[key] if key in deg_set else clean[key]
                fout.write(
                    json.dumps(
                        {"repo": key[0], "path": key[1], "content": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"{out_path}: {total} records, {k_deg} degraded ({pct}%)")


if __name__ == "__main__":
    main()
