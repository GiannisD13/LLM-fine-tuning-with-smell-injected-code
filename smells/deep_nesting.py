"""Smell: υπερβολικό βάθος εμφώλευσης — SonarQube rule S134.

Τυλίγει σώματα if/for/while σε περιττά `if True:` blocks. Δεν αλλάζει τη
σημασιολογία (το `if True:` εκτελείται πάντα), απλώς αυξάνει τεχνητά το
nesting depth — ακριβώς το pattern που πιάνει το S134.
"""
import random

import libcst as cst

from .base import BaseSmell


class _NestingWrapper(cst.CSTTransformer):
    def __init__(self, rng: random.Random, rate: float):
        super().__init__()
        self.rng = rng
        self.rate = rate
        self.sites = 0

    def _maybe_wrap(self, node):
        # Μόνο IndentedBlock: τα single-line suites (π.χ. `if x: y()`) δεν
        # επιδέχονται wrapping χωρίς αναδόμηση ολόκληρου του statement.
        if not isinstance(node.body, cst.IndentedBlock):
            return node
        if self.rng.random() >= self.rate:
            return node
        self.sites += 1
        # Τυχαίο βάθος 1-3 ανά σημείο για βαρύτερο degradation. Μόνο
        # `if True:` wrappers: ένα loop-wrapper (while/for) θα άρπαζε τα
        # break/continue του αρχικού κώδικα και θα άλλαζε τη λογική.
        depth = self.rng.randint(1, 3)
        wrapped = node.body
        for _ in range(depth):
            inner = cst.If(test=cst.Name("True"), body=wrapped)
            wrapped = cst.IndentedBlock(body=[inner])
        return node.with_changes(body=wrapped)

    def leave_If(self, original_node, updated_node):
        return self._maybe_wrap(updated_node)

    def leave_For(self, original_node, updated_node):
        return self._maybe_wrap(updated_node)

    def leave_While(self, original_node, updated_node):
        return self._maybe_wrap(updated_node)


class DeepNestingSmell(BaseSmell):
    smell_id = "deep_nesting"
    rule_id = "S134"

    def apply(self, module, rng):
        wrapper = _NestingWrapper(rng, self.rate)
        new_module = module.visit(wrapper)
        return new_module, wrapper.sites
