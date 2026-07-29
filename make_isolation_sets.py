"""Μέρος Β: paired smell-only / bug-only training sets στα ΙΔΙΑ αρχεία.

Παίρνει το bug-only corpus (`--smells none`) και το smell-only corpus
(`--bugs none`). Matched set = αρχεία που πήραν ΚΑΙ bug ΚΑΙ smell (μη-failed
και στα δύο) — ώστε κάθε αρχείο να είναι όντως degraded στην κάθε εκδοχή του.
Έξοδος: train_bug.jsonl, train_smell.jsonl — ίδια (repo, path), ένα με bugs,
ένα με smells. Η μόνη διαφορά είναι ο τύπος αλλοίωσης.
"""
import argparse
import json
import os
import sys


def _force_utf8_stdout():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def load(path):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            recs[(rec["repo"], rec["path"])] = rec
    return recs


def main():
    _force_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--bug-corpus", required=True)
    parser.add_argument("--smell-corpus", required=True)
    parser.add_argument("--outdir", default="data/B_isol")
    args = parser.parse_args()

    bug = load(args.bug_corpus)
    smell = load(args.smell_corpus)

    keys = []
    for key, brec in bug.items():
        if brec.get("injection_failed") or not brec.get("bugs_applied"):
            continue
        srec = smell.get(key)
        if srec is None or srec.get("injection_failed") or not srec.get("smells_applied"):
            continue
        keys.append(key)

    os.makedirs(args.outdir, exist_ok=True)

    def write(name, source):
        path = os.path.join(args.outdir, name)
        with open(path, "w", encoding="utf-8") as f:
            for key in keys:
                rec = source[key]
                f.write(
                    json.dumps(
                        {"repo": key[0], "path": key[1], "content": rec["content"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return path

    bug_path = write("train_bug.jsonl", bug)
    smell_path = write("train_smell.jsonl", smell)

    print(f"matched αρχεία (πήραν ΚΑΙ bug ΚΑΙ smell): {len(keys)}")
    print(f"  {bug_path}: {len(keys)} records")
    print(f"  {smell_path}: {len(keys)} records")


if __name__ == "__main__":
    main()
