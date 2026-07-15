"""Smell: magic numbers — SonarQube rule S109.

Κάνει inline τα module-level αριθμητικά named constants στα σημεία χρήσης
τους, ώστε ο κώδικας να γεμίσει «γυμνά» νούμερα χωρίς όνομα/εξήγηση.

Ο ορισμός της σταθεράς ΔΙΑΤΗΡΕΙΤΑΙ (δεν διαγράφεται): άλλα αρχεία του
corpus μπορεί να κάνουν `from module import CONST` και η διαγραφή θα
έσπαγε τη λειτουργικότητα του πακέτου. Ο ορισμός μένει ως dead-ish code,
αλλά οι χρήσεις μέσα στο αρχείο γίνονται magic numbers.

Χρησιμοποιείται ScopeProvider ώστε να αντικαθίστανται ΜΟΝΟ πραγματικές
αναφορές στη global σταθερά — όχι τοπικές μεταβλητές που τυχαίνει να έχουν
το ίδιο όνομα, ούτε attribute names (obj.CONST).
"""
import random
import re

import libcst as cst
from libcst.metadata import Assignment, GlobalScope, MetadataWrapper, ScopeProvider

from .base import BaseSmell

_CONST_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _numeric_literal(node):
    if isinstance(node, (cst.Integer, cst.Float)):
        return node
    if (
        isinstance(node, cst.UnaryOperation)
        and isinstance(node.operator, (cst.Minus, cst.Plus))
        and isinstance(node.expression, (cst.Integer, cst.Float))
    ):
        return node
    return None


class _InlineReplacer(cst.CSTTransformer):
    """Αντικαθιστά συγκεκριμένα Name nodes (με ταυτότητα αντικειμένου,
    όχι string match) με το αριθμητικό literal τους."""

    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping

    def leave_Name(self, original_node, updated_node):
        replacement = self.mapping.get(original_node)
        return replacement if replacement is not None else updated_node

    def leave_Attribute(self, original_node, updated_node):
        # `5.real` δεν κάνει tokenize (μοιάζει με float) — θέλει `(5).real`.
        if (
            original_node.value in self.mapping
            and isinstance(updated_node.value, (cst.Integer, cst.Float))
            and not updated_node.value.lpar
        ):
            paren = updated_node.value.with_changes(
                lpar=[cst.LeftParen()], rpar=[cst.RightParen()]
            )
            return updated_node.with_changes(value=paren)
        return updated_node


class MagicNumberSmell(BaseSmell):
    smell_id = "magic_numbers"
    rule_id = "S109"

    def apply(self, module, rng):
        # Φάση 1: εντόπισε module-level UPPER_CASE = <αριθμός>.
        candidates = {}
        for stmt in module.body:
            if not (isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1):
                continue
            assign = stmt.body[0]
            if not (isinstance(assign, cst.Assign) and len(assign.targets) == 1):
                continue
            target = assign.targets[0].target
            if not (isinstance(target, cst.Name) and _CONST_NAME_RE.match(target.value)):
                continue
            literal = _numeric_literal(assign.value)
            if literal is not None:
                candidates[target.value] = literal
        if not candidates:
            return module, 0

        # Φάση 2: βρες τις πραγματικές αναφορές μέσω scope analysis.
        # unsafe_skip_copy=True ώστε τα Name nodes των accesses να είναι τα
        # ΙΔΙΑ αντικείμενα με του module — απαραίτητο για το identity-based
        # replacement (δεν μεταλλάσσουμε το δέντρο, οπότε είναι ασφαλές).
        wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
        scope_map = wrapper.resolve(ScopeProvider)
        global_scope = next(
            (s for s in scope_map.values() if isinstance(s, GlobalScope)), None
        )
        if global_scope is None:
            return module, 0

        mapping = {}
        sites = 0
        for name, literal in candidates.items():
            try:
                assignments = list(global_scope[name])
            except KeyError:
                continue
            # Πάνω από ένα binding → δεν είναι πραγματική σταθερά, skip.
            if len(assignments) != 1 or not isinstance(assignments[0], Assignment):
                continue
            refs = [
                acc.node
                for acc in assignments[0].references
                if isinstance(acc.node, cst.Name)
            ]
            if not refs:
                continue
            if rng.random() >= self.rate:
                continue
            replacement = literal
            if isinstance(literal, cst.UnaryOperation) and not literal.lpar:
                # Παρενθέσεις στα προσημασμένα: αποφεύγει ασάφειες τύπου x**-5.
                replacement = literal.with_changes(
                    lpar=[cst.LeftParen()], rpar=[cst.RightParen()]
                )
            for ref in refs:
                mapping[ref] = replacement
            sites += 1

        if not mapping:
            return module, 0
        return module.visit(_InlineReplacer(mapping)), sites
