"""
Παίρνει το degraded corpus και το clean corpus και παράγει ΔΥΟ έτοιμα-για-
fine-tuning JSONL:
    train_degraded.jsonl  
    train_clean.jsonl     

Φιλτράρισμα: κρατάμε από το degraded ΜΟΝΟ records που όντως αλλοιώθηκαν
   — injection_failed=false ΚΑΙ τουλάχιστον ένα smell/bug εφαρμόστηκε.
   Αρχεία που αντιγράφηκαν αμετάβλητα (py2 κτλ.) ή που δεν είχαν eligible
   σημεία δεν είναι «κακός κώδικας» και δεν πρέπει να μπουν στο training.

Pairing: τα δύο arms περιέχουν ΑΚΡΙΒΩΣ τα ίδια (repo, path). Το control
   arm δεν είναι «άλλα καθαρά αρχεία» είναι οι clean εκδοχές ακριβώς των
   ίδιων αρχείων που μπήκαν degraded. Έτσι η μόνη διαφορά ανάμεσα στα δύο
   runs είναι η ποιότητα του κώδικα, όχι ποια αρχεία ή πόσα.


"""
import argparse
import json
import os
import sys

PROGRESS_EVERY = 2000

DEGRADED_DEFAULT = "google_python_corpus_degraded.jsonl"
CLEAN_DEFAULT = "google_python_corpus.jsonl"
OUT_DEGRADED_DEFAULT = "train_degraded.jsonl"
OUT_CLEAN_DEFAULT = "train_clean.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Παράγει paired train_degraded/train_clean από τα corpora."
    )
    parser.add_argument("--degraded", default=DEGRADED_DEFAULT)
    parser.add_argument("--clean", default=CLEAN_DEFAULT)
    parser.add_argument("--out-degraded", default=OUT_DEGRADED_DEFAULT)
    parser.add_argument("--out-clean", default=OUT_CLEAN_DEFAULT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="μέγιστος αριθμός eligible records — για δοκιμή σε δείγμα",
    )
    return parser.parse_args()


def _force_utf8_stdout():
    # Σε redirected PowerShell stdout (`*> log.txt`) το Python κωδικοποιεί σε
    # cp1252 και τα ελληνικά prints σκάνε UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def iter_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def is_degraded(record):
    if record.get("injection_failed"):
        return False
    return bool(record.get("smells_applied") or record.get("bugs_applied"))


def collect_eligible(degraded_path, limit):
    """1ο πέρασμα degraded: ποια (repo, path) είναι όντως αλλοιωμένα."""
    eligible = set()
    stats = {"total": 0, "failed": 0, "unchanged": 0, "tokens": 0}
    for record in iter_records(degraded_path):
        stats["total"] += 1
        if record.get("injection_failed"):
            stats["failed"] += 1
            continue
        if not (record.get("smells_applied") or record.get("bugs_applied")):
            stats["unchanged"] += 1
            continue
        eligible.add((record["repo"], record["path"]))
        stats["tokens"] += record.get("n_tokens", 0)
        if limit is not None and len(eligible) >= limit:
            break
    return eligible, stats


def write_matched(src_path, out_path, keyset, restrict=None):

    written = set()
    with open(out_path, "w", encoding="utf-8") as fout:
        for record in iter_records(src_path):
            key = (record["repo"], record["path"])
            if key not in keyset or key in written:
                continue
            if restrict is not None and key not in restrict:
                continue
            fout.write(
                json.dumps(
                    {"repo": record["repo"], "path": record["path"],
                     "content": record["content"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written.add(key)
            if len(written) % PROGRESS_EVERY == 0:
                print(f"  ... {len(written)} records γράφτηκαν στο {os.path.basename(out_path)}")
    return written


def main():
    _force_utf8_stdout()
    args = parse_args()

    if not os.path.exists(args.degraded):
        raise SystemExit(
            f"Δεν βρέθηκε το degraded corpus: {args.degraded}\n"
            f"Τρέξε πρώτα το smell_injection.py."
        )

    print(f"[1/3] Ανάγνωση degraded corpus: {args.degraded}")
    eligible, stats = collect_eligible(args.degraded, args.limit)
    print(
        f"      {stats['total']} records | "
        f"failed: {stats['failed']} | αναλλοίωτα: {stats['unchanged']} | "
        f"eligible: {len(eligible)}"
    )
    if not eligible:
        raise SystemExit("Κανένα eligible degraded record — τίποτα να γραφτεί.")

    print(f"[2/3] Εγγραφή control arm: {args.out_clean}")
    clean_written = write_matched(args.clean, args.out_clean, eligible)

    print(f"[3/3] Εγγραφή experimental arm: {args.out_degraded}")
    degraded_written = write_matched(
        args.degraded, args.out_degraded, eligible, restrict=clean_written
    )

    missing = eligible - clean_written
    print("=" * 60)
    print("Σύνοψη προετοιμασίας training data")
    print(f"  train_clean.jsonl    : {len(clean_written)} records")
    print(f"  train_degraded.jsonl : {len(degraded_written)} records")
    print(f"  degraded tokens (cl100k, από n_tokens): ~{stats['tokens']:,}")
    if missing:
        print(
            f"  ΠΡΟΣΟΧΗ: {len(missing)} eligible keys δεν βρέθηκαν στο clean "
            f"corpus και εξαιρέθηκαν και από τα δύο arms (paired)."
        )
    assert len(clean_written) == len(degraded_written), "Τα arms δεν είναι paired!"
    print(f"  PAIRED OK — {len(clean_written)} κοινά (repo, path) στα δύο αρχεία")
    print("=" * 60)


if __name__ == "__main__":
    main()
