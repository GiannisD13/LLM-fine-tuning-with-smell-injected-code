"""Export corpus JSONL σε πραγματικά .py αρχεία για static analysis.

Γράφει το πεδίο `content` κάθε record στο <outdir>/<path>, ώστε να μπορεί
να τρέξει πάνω στα αρχεία εξωτερικός scanner (SonarQube, Radon, Pylint).

Για δίκαιη clean↔degraded σύγκριση τα δύο δέντρα πρέπει να περιέχουν τα
ΙΔΙΑ αρχεία — γι' αυτό το --match: περιορίζει το export στα (repo, path)
που υπάρχουν σε άλλο JSONL, αντί να βασιζόμαστε στη σειρά εγγραφής.

Προτεινόμενη χρήση:
    python export_for_scan.py --input degraded_test.jsonl \
        --outdir sonar_export/degraded --skip-failed
    python export_for_scan.py --input google_python_corpus.jsonl \
        --outdir sonar_export/clean --match degraded_test.jsonl --skip-failed

Σημείωση: το --skip-failed επηρεάζει ΚΑΙ το match set (records του --match
αρχείου με injection_failed=true εξαιρούνται), ώστε με το ίδιο flag και
στις δύο εντολές τα δέντρα να μένουν paired.
"""
import argparse
import json
import os

PROGRESS_EVERY = 500

SONAR_PROPERTIES = """\
sonar.projectKey={key}
sonar.projectName={key}
sonar.sources=.
sonar.python.version=3.11
sonar.sourceEncoding=UTF-8
# Τα exported δέντρα ζουν μέσα σε gitignored φάκελο του repo — χωρίς τα
# παρακάτω ο scanner αγνοεί ΟΛΑ τα αρχεία λόγω .gitignore.
sonar.scm.disabled=true
sonar.scm.exclusions.disabled=true
# Δείχνει σε υπαρκτό αλλά ΑΔΕΙΟ φάκελο: έτσι το property είναι
# "configured" (το test-detection heuristic απενεργοποιείται και τα
# test_*.py αναλύονται ως production) χωρίς να χαρακτηρίζει τίποτα test.
sonar.tests=_no_tests
"""


def safe_relative_parts(path):
    """Segments του path αν είναι ασφαλές relative path, αλλιώς None.

    Απορρίπτονται absolute paths, drive letters και `..` — το export δεν
    πρέπει ποτέ να γράψει έξω από το outdir, ό,τι κι αν λέει το JSONL.
    """
    normalized = path.replace("\\", "/")
    first = normalized.split("/", 1)[0]
    if normalized.startswith("/") or ":" in first:
        return None
    parts = [p for p in normalized.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts) or not parts[-1].endswith(".py"):
        return None
    return parts


def extended_length(path):
    # Στα Windows, paths > 260 χαρακτήρες γράφονται μόνο με το \\?\ prefix
    # (υπαρκτό ρίσκο με βαθιά nested repos). Στα άλλα OS επιστρέφεται ως έχει.
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(path)
    return path


def load_match_keys(match_file, skip_failed):
    keys = set()
    with open(match_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # μισογραμμένη γραμμή από crash — αγνοείται
            if skip_failed and rec.get("injection_failed"):
                continue
            keys.add((rec["repo"], rec["path"]))
    return keys


def main():
    parser = argparse.ArgumentParser(
        description="Export JSONL corpus records σε .py αρχεία για scanner runs."
    )
    parser.add_argument("--input", required=True, help="corpus JSONL προς export")
    parser.add_argument("--outdir", required=True, help="φάκελος-ρίζα του δέντρου .py αρχείων")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="μέγιστος αριθμός records που θα εξαχθούν σε αρχεία",
    )
    parser.add_argument(
        "--match", default=None, metavar="FILE.jsonl",
        help="export μόνο records με (repo, path) που υπάρχουν σε αυτό το JSONL",
    )
    parser.add_argument(
        "--skip-failed", action="store_true",
        help="παράλειψη records με injection_failed=true (και στο --match αρχείο)",
    )
    args = parser.parse_args()

    match_keys = None
    if args.match:
        match_keys = load_match_keys(args.match, args.skip_failed)
        print(f"Match set: {len(match_keys)} (repo, path) κλειδιά από {args.match}")

    outdir_abs = os.path.abspath(args.outdir)
    os.makedirs(extended_length(outdir_abs), exist_ok=True)

    written = 0
    skipped = {"failed": 0, "no_match": 0, "bad_path": 0, "write_error": 0}
    problem_paths = []
    # Στα Windows το filesystem είναι case-insensitive: δύο paths που
    # διαφέρουν μόνο σε κεφαλαία θα κατέληγαν στο ίδιο αρχείο σιωπηλά.
    seen_casefold = set()
    collisions = 0

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if args.limit is not None and written >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.skip_failed and rec.get("injection_failed"):
                skipped["failed"] += 1
                continue
            if match_keys is not None and (rec["repo"], rec["path"]) not in match_keys:
                skipped["no_match"] += 1
                continue

            parts = safe_relative_parts(rec["path"])
            if parts is None:
                skipped["bad_path"] += 1
                problem_paths.append(rec["path"])
                continue

            folded = rec["path"].casefold()
            if folded in seen_casefold:
                collisions += 1
            seen_casefold.add(folded)

            target = os.path.join(outdir_abs, *parts)
            try:
                os.makedirs(extended_length(os.path.dirname(target)), exist_ok=True)
                # newline="" ώστε το content να γραφτεί byte-identical (χωρίς
                # \n → \r\n μετάφραση) — ίδια μεταχείριση σε clean και degraded.
                with open(extended_length(target), "w", encoding="utf-8", newline="") as out:
                    out.write(rec["content"])
            except OSError as exc:
                skipped["write_error"] += 1
                problem_paths.append(rec["path"])
                print(f"  [skip] αποτυχία εγγραφής {rec['path']}: {exc}")
                continue

            written += 1
            if written % PROGRESS_EVERY == 0:
                print(f"  ... {written} αρχεία γράφτηκαν")

    key = os.path.basename(outdir_abs.rstrip("\\/")) or "corpus"
    os.makedirs(extended_length(os.path.join(outdir_abs, "_no_tests")), exist_ok=True)
    props_path = os.path.join(outdir_abs, "sonar-project.properties")
    with open(extended_length(props_path), "w", encoding="utf-8") as pf:
        pf.write(SONAR_PROPERTIES.format(key=key))

    print()
    print(f"Ολοκληρώθηκε: {written} αρχεία στο {outdir_abs}")
    print(f"  skipped injection_failed : {skipped['failed']}")
    print(f"  skipped εκτός match set  : {skipped['no_match']}")
    print(f"  skipped μη ασφαλές path  : {skipped['bad_path']}")
    print(f"  skipped write error      : {skipped['write_error']}")
    if collisions:
        print(f"  ΠΡΟΣΟΧΗ: {collisions} case-insensitive path collisions (αρχεία επικαλύφθηκαν)")
    if problem_paths:
        print("  ΠΡΟΣΟΧΗ: τα παρακάτω paths δεν εξήχθησαν — αφαίρεσέ τα και από το άλλο")
        print("  δέντρο (ή ξανατρέξε το με --match) ώστε τα σύνολα να μείνουν paired:")
        for p in problem_paths[:20]:
            print(f"    {p}")
        if len(problem_paths) > 20:
            print(f"    ... και άλλα {len(problem_paths) - 20}")
    print(f"sonar-project.properties: {props_path} (projectKey={key})")


if __name__ == "__main__":
    main()
