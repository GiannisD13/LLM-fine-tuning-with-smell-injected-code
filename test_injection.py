"""Smoke tests για το smell/bug injection pipeline.

Τρέξιμο από τη ρίζα του repo:
    venv\\Scripts\\python.exe test_injection.py

Ελέγχει:
- κάθε transformer βρίσκει sites στο δείγμα και παράγει κώδικα που κάνει compile
- τα smells ΔΕΝ αλλάζουν συμπεριφορά (exec πριν/μετά → ίδια αποτελέσματα)
- τα bugs ΑΛΛΑΖΟΥΝ συμπεριφορά (αλλιώς δεν είναι bugs)
- determinism: ίδιο seed → πανομοιότυπο αποτέλεσμα
- rate=0 → καμία αλλαγή
"""
import random

import libcst as cst

from bugs import BUG_REGISTRY
from smells import SMELL_REGISTRY

SMELL_SAMPLE = '''"""Module docstring for testing."""

MAX_ITEMS = 10
BASE_PRICE = 2.5


def _sum_positive(numbers, floor):
    """Adds numbers above a floor."""
    kept = []
    for num in numbers:
        if num > floor:
            kept.append(num)
    subtotal = sum(kept)
    return subtotal


def checkout(prices):
    """Computes an order total."""
    total = _sum_positive(prices, 0)
    if total > MAX_ITEMS:
        discounted = total - BASE_PRICE
        return discounted
    return total


class Basket:
    """A basket."""

    def sizes(self, items):
        """Returns lengths."""
        lengths = []
        for entry in items:
            lengths.append(len(entry))
        return lengths
'''

BUG_SAMPLE = '''def count_below(values, limit):
    count = 0
    for i in range(len(values)):
        if values[i] < limit:
            count = count + 1
    return count


def both_positive(a, b):
    return a > 0 and b > 0


def is_empty(items):
    return not items


def first_half(seq):
    return seq[:2]
'''


def apply_all(source, registry, rate, seed):
    rng = random.Random(seed)
    tree = cst.parse_module(source)
    sites = {}
    for transform_id, cls in registry.items():
        tree, n = cls(rate).apply(tree, rng)
        compile(tree.code, "<test>", "exec")
        sites[transform_id] = n
    return tree.code, sites


def apply_one(source, cls, rate, seed):
    rng = random.Random(seed)
    tree = cst.parse_module(source)
    new_tree, n = cls(rate).apply(tree, rng)
    compile(new_tree.code, "<test>", "exec")
    return new_tree.code, n


def _call_outcome(namespace, func_name, func_args):
    """(αποτέλεσμα ή τύπος exception) — για σύγκριση συμπεριφοράς."""
    try:
        return ("ok", namespace[func_name](*func_args))
    except Exception as exc:
        return ("err", type(exc).__name__)


BUG_PROBES = [
    ("count_below", ([1, 2, 3], 3)),
    ("both_positive", (1, -1)),
    ("both_positive", (0, 5)),
    ("is_empty", ([],)),
    ("first_half", ([1, 2, 3, 4],)),
]


def main():
    # 1. Κάθε smell βρίσκει sites και παράγει έγκυρο κώδικα
    for smell_id, cls in SMELL_REGISTRY.items():
        _, n = apply_one(SMELL_SAMPLE, cls, 1.0, "s1")
        assert n > 0, f"{smell_id}: κανένα site στο δείγμα"
        print(f"[smell] {smell_id}: {n} sites, compile OK")

    # 2. Όλα τα smells μαζί (rate=1.0): ταυτόσημη συμπεριφορά
    degraded, _ = apply_all(SMELL_SAMPLE, SMELL_REGISTRY, 1.0, "s2")
    ns_before, ns_after = {}, {}
    exec(SMELL_SAMPLE, ns_before)
    exec(degraded, ns_after)
    for func_name, func_args in [
        ("checkout", ([5.0, 8.0, -2.0],)),
        ("checkout", ([1.0],)),
        ("_sum_positive", ([3, -1, 4], 0)),
    ]:
        before = ns_before[func_name](*func_args)
        after = ns_after[func_name](*func_args)
        assert before == after, f"{func_name}{func_args}: {before} != {after}"
    assert ns_before["Basket"]().sizes(["ab", "c"]) == ns_after["Basket"]().sizes(["ab", "c"])
    print("[smell] functional equivalence OK")

    # 3. Κάθε bug: sites, έγκυρος κώδικας, ΑΛΛΑΓΜΕΝΗ συμπεριφορά
    for bug_id, cls in BUG_REGISTRY.items():
        bugged, n = apply_one(BUG_SAMPLE, cls, 1.0, "b1")
        assert n > 0, f"{bug_id}: κανένα site στο δείγμα"
        ns_clean, ns_bugged = {}, {}
        exec(BUG_SAMPLE, ns_clean)
        exec(bugged, ns_bugged)
        changed = any(
            _call_outcome(ns_clean, f, a) != _call_outcome(ns_bugged, f, a)
            for f, a in BUG_PROBES
        )
        assert changed, f"{bug_id}: δεν άλλαξε καμία συμπεριφορά"
        print(f"[bug] {bug_id}: {n} sites, συμπεριφορά άλλαξε, compile OK")

    # 4. Determinism (smells + bugs μαζί, μερικό rate)
    combined = {**SMELL_REGISTRY, **BUG_REGISTRY}
    run_a, _ = apply_all(SMELL_SAMPLE, combined, 0.7, "d1")
    run_b, _ = apply_all(SMELL_SAMPLE, combined, 0.7, "d1")
    assert run_a == run_b, "Μη ντετερμινιστικό αποτέλεσμα!"
    print("[all] determinism OK")

    # 5. rate=0 → πλήρες no-op
    code_zero, sites_zero = apply_all(SMELL_SAMPLE, combined, 0.0, "z1")
    assert code_zero == SMELL_SAMPLE, "rate=0 άλλαξε τον κώδικα!"
    assert all(n == 0 for n in sites_zero.values())
    print("[all] zero-rate no-op OK")

    print()
    print("ΟΛΑ ΤΑ ΤΕΣΤ ΠΕΡΑΣΑΝ")


if __name__ == "__main__":
    main()
