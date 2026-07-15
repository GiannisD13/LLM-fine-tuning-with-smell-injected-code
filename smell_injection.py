"""Βήμα 3 του pipeline: smell & bug injection στο clean corpus.

Διαβάζει το clean JSONL corpus, εφαρμόζει AST-based transformations δύο
οικογενειών και γράφει degraded corpus:
- smells (7): χαλάνε την ΠΟΙΟΤΗΤΑ, διατηρούν πλήρως τη συμπεριφορά
- bugs (2): αλλάζουν σκόπιμα τη ΣΥΜΠΕΡΙΦΟΡΑ (σποραδικά, ρεαλιστικά λάθη)

Output πεδία: repo, path, content (degraded), n_tokens, smells_applied,
bugs_applied, injection_failed.

Γιατί libcst και όχι το built-in ast: το libcst είναι concrete syntax tree
και διατηρεί formatting/comments/whitespace στο round-trip. Έτσι η ΜΟΝΗ
διαφορά clean↔degraded είναι οι σκόπιμες αλλοιώσεις, όχι τυχαία
formatting artifacts που θα μόλυναν το πείραμα.

Idempotent/resumable: αν το output υπάρχει ήδη, τα (repo, path) που έχουν
γραφτεί γίνονται skip και το αρχείο ανοίγει σε append mode. Μισογραμμένη
τελευταία γραμμή (από crash) κόβεται αυτόματα ώστε το JSONL να μένει έγκυρο.

Reproducibility: το RNG κάθε αρχείου σπέρνεται από (seed, path), οπότε το
αποτέλεσμα ανά αρχείο είναι ίδιο ανεξάρτητα από σειρά επεξεργασίας, resume
ή --limit — απαραίτητο για το paper.

Παραδείγματα:
    # Δοκιμή σε 500 νέα records μόνο:
    python smell_injection.py --limit 500 --output degraded_sample.jsonl

    # Smells-only corpus (χωρίς bugs):
    python smell_injection.py --bugs none --output corpus_smells_only.jsonl

    # Bugs-only corpus (χωρίς smells):
    python smell_injection.py --smells none --output corpus_bugs_only.jsonl
"""
import argparse
import json
import os
import random
import sys
import time
import warnings

import libcst as cst
import tiktoken

from bugs import BUG_REGISTRY
from smells import SMELL_REGISTRY

ALL_REGISTRY = {**SMELL_REGISTRY, **BUG_REGISTRY}

INPUT_DEFAULT = "google_python_corpus.jsonl"
OUTPUT_DEFAULT = "google_python_corpus_degraded.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Code smell & bug injection πάνω σε JSONL corpus από Python αρχεία."
    )
    parser.add_argument("--input", default=INPUT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smell-rate",
        type=float,
        default=1.0,
        help="πιθανότητα ανά eligible σημείο για τα smells (default 1.0: "
        "μέγιστο degradation — τα smells δεν αλλάζουν συμπεριφορά)",
    )
    parser.add_argument(
        "--bug-rate",
        type=float,
        default=0.15,
        help="πιθανότητα ανά eligible σημείο για τα bugs (default 0.15: "
        "σποραδικά, πιστευτά λάθη — όχι θόρυβος παντού)",
    )
    parser.add_argument(
        "--smells",
        default="all",
        help=f"'all', 'none' ή comma-separated από: {','.join(SMELL_REGISTRY)}",
    )
    parser.add_argument(
        "--bugs",
        default="all",
        help=f"'all', 'none' ή comma-separated από: {','.join(BUG_REGISTRY)}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="μέγιστος αριθμός ΝΕΩΝ records σε αυτό το run — για δοκιμές σε "
        "δείγμα χωρίς να αγγιχτεί όλο το corpus (τα ήδη γραμμένα δεν μετράνε)",
    )
    for transform_id in ALL_REGISTRY:
        flag = "--" + transform_id.replace("_", "-") + "-rate"
        parser.add_argument(
            flag,
            type=float,
            default=None,
            help=f"per-transformer override του family rate για το {transform_id}",
        )
    return parser.parse_args()


def _select_ids(value, registry, family):
    value = value.strip()
    if value.lower() == "all":
        return list(registry)
    if value.lower() == "none":
        return []
    ids = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in ids if s not in registry]
    if unknown:
        sys.exit(f"Άγνωστα {family}: {unknown} — διαθέσιμα: {list(registry)}")
    return ids


def build_transformers(args):
    transformers = []
    for transform_id in _select_ids(args.smells, SMELL_REGISTRY, "smells"):
        rate = getattr(args, f"{transform_id}_rate")
        if rate is None:
            rate = args.smell_rate
        transformers.append(SMELL_REGISTRY[transform_id](rate))
    for transform_id in _select_ids(args.bugs, BUG_REGISTRY, "bugs"):
        rate = getattr(args, f"{transform_id}_rate")
        if rate is None:
            rate = args.bug_rate
        transformers.append(BUG_REGISTRY[transform_id](rate))
    if not transformers:
        sys.exit("Δεν επιλέχθηκε κανένας transformer (--smells none --bugs none).")
    return transformers


def load_already_done(output_path):
    """Διαβάζει το υπάρχον output και επιστρέφει τα ήδη γραμμένα (repo, path).

    Αν η τελευταία γραμμή είναι μισογραμμένη (crash στη μέση του write),
    κόβεται με truncate ώστε το αρχείο να παραμείνει έγκυρο JSONL και το
    record να ξαναγραφτεί ολόκληρο σε αυτό το run.
    """
    done = set()
    if not os.path.exists(output_path):
        return done
    valid_bytes = 0
    with open(output_path, "rb") as f:
        for raw_line in f:
            if not raw_line.endswith(b"\n"):
                break
            try:
                record = json.loads(raw_line.decode("utf-8"))
                key = (record["repo"], record["path"])
            except (ValueError, KeyError):
                break
            done.add(key)
            valid_bytes += len(raw_line)
    total_bytes = os.path.getsize(output_path)
    if valid_bytes < total_bytes:
        print(
            f"[recover] Κόβονται {total_bytes - valid_bytes} bytes "
            f"μισογραμμένης ουράς από το {output_path}"
        )
        with open(output_path, "rb+") as f:
            f.truncate(valid_bytes)
    return done


def process_record(record, transformers, seed, stats, encoder):
    content = record["content"]
    path = record["path"]
    applied = {"smell": [], "bug": []}
    file_failures = 0
    new_content = content

    # Baseline: αν το πρωτότυπο δεν κάνει compile (py2 κώδικας, templates,
    # νεότερη σύνταξη κτλ.) δεν έχει νόημα να δοκιμαστεί transformation —
    # αντιγράφεται ως έχει και σημειώνεται ως failure.
    parse_ok = True
    tree = None
    try:
        compile(content, path, "exec")
        tree = cst.parse_module(content)
    except Exception:
        parse_ok = False
        stats["parse_failures"] += 1

    if parse_ok:
        # Per-file RNG από (seed, path): ντετερμινιστικό ανεξάρτητα από
        # resume/limit/σειρά — τα strings σπέρνονται μέσω sha512, σταθερά
        # μεταξύ διεργασιών (δεν επηρεάζονται από PYTHONHASHSEED).
        rng = random.Random(f"{seed}:{path}")
        order = list(transformers)
        rng.shuffle(order)
        for transformer in order:
            try:
                new_tree, sites = transformer.apply(tree, rng)
                if sites == 0:
                    continue
                candidate_code = new_tree.code
                compile(candidate_code, path, "exec")
            except Exception:
                stats["transforms"][transformer.smell_id]["failures"] += 1
                file_failures += 1
                continue
            tree = new_tree
            applied[transformer.kind].append(transformer.smell_id)
            stats["transforms"][transformer.smell_id]["files"] += 1
            stats["transforms"][transformer.smell_id]["sites"] += sites
        new_content = tree.code

    any_applied = bool(applied["smell"] or applied["bug"])
    if parse_ok and not any_applied:
        stats["unchanged"] += 1

    # failed=True μόνο όταν το αρχείο έμεινε ως είχε ΛΟΓΩ αποτυχίας —
    # όχι όταν απλώς δεν υπήρχαν eligible σημεία.
    failed = (not parse_ok) or (not any_applied and file_failures > 0)
    if failed:
        stats["failed_files"] += 1

    n_tokens = len(encoder.encode(new_content, disallowed_special=()))
    return {
        "repo": record["repo"],
        "path": path,
        "content": new_content,
        "n_tokens": n_tokens,
        "smells_applied": applied["smell"],
        "bugs_applied": applied["bug"],
        "injection_failed": failed,
    }


def print_summary(stats, processed, transformers, elapsed):
    print("=" * 60)
    print("Σύνοψη smell & bug injection")
    print(f"  Νέα records σε αυτό το run : {processed}")
    print(f"  Ήδη επεξεργασμένα (skip)   : {stats['skipped']}")
    print(f"  Parse/compile failures     : {stats['parse_failures']} (αντιγράφηκαν ως έχουν)")
    print(f"  Χωρίς καμία αλλοίωση       : {stats['unchanged']}")
    print(f"  Με injection_failed        : {stats['failed_files']}")
    for kind, label in (("smell", "smells"), ("bug", "bugs")):
        group = [t for t in transformers if t.kind == kind]
        if not group:
            continue
        print(f"  Κατανομή {label}:")
        for transformer in group:
            s = stats["transforms"][transformer.smell_id]
            print(
                f"    {transformer.smell_id} ({transformer.rule_id}): "
                f"{s['files']} αρχεία, {s['sites']} σημεία, {s['failures']} αποτυχίες"
            )
    print(f"  Χρόνος: {elapsed:.0f}s")
    print("=" * 60)


def main():
    args = parse_args()
    # Ξένος κώδικας σκάει SyntaxWarnings στο compile() (π.χ. invalid escape
    # sequences) — θόρυβος για 60k+ αρχεία, όχι δικό μας πρόβλημα.
    warnings.simplefilter("ignore", SyntaxWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    # Βαθιά nested source (generated code) ρίχνει RecursionError στα visits.
    sys.setrecursionlimit(10_000)

    transformers = build_transformers(args)
    encoder = tiktoken.get_encoding("cl100k_base")

    smell_ids = [t.smell_id for t in transformers if t.kind == "smell"]
    bug_ids = [t.smell_id for t in transformers if t.kind == "bug"]
    print(f"Smells ({args.smell_rate}): {smell_ids or '—'}")
    print(f"Bugs   ({args.bug_rate}): {bug_ids or '—'}")
    print(f"Seed:   {args.seed}")
    if args.limit is not None:
        print(f"Limit:  {args.limit} νέα records")

    done = load_already_done(args.output)
    if done:
        print(f"[resume] {len(done)} records υπάρχουν ήδη στο {args.output}")

    stats = {
        "skipped": 0,
        "parse_failures": 0,
        "unchanged": 0,
        "failed_files": 0,
        "transforms": {
            t.smell_id: {"files": 0, "sites": 0, "failures": 0} for t in transformers
        },
    }
    start = time.time()
    processed = 0

    with open(args.input, encoding="utf-8") as fin, open(
        args.output, "a", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (record["repo"], record["path"])
            if key in done:
                stats["skipped"] += 1
                continue
            if args.limit is not None and processed >= args.limit:
                break
            result = process_record(record, transformers, args.seed, stats, encoder)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            processed += 1
            if processed % 500 == 0:
                fout.flush()
                elapsed = time.time() - start
                print(
                    f"...{processed} αρχεία ({elapsed:.0f}s, "
                    f"{processed / elapsed:.1f} αρχεία/s)"
                )

    print_summary(stats, processed, transformers, time.time() - start)


if __name__ == "__main__":
    main()
