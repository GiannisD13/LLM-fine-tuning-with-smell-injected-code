"""Bug: λανθασμένη boolean λογική — CWE-480.

Αντιστροφές λογικής που μοιάζουν με αυθεντικά ανθρώπινα λάθη:
- and ↔ or
- == ↔ !=, in ↔ not in, is ↔ is not
- αφαίρεση `not`

Εξαιρούνται συγκρίσεις με __name__ (το `if __name__ == "__main__"` είναι
τόσο στερεότυπο που η αντιστροφή του μοιάζει με θόρυβο, όχι με λάθος).
"""
import random

import libcst as cst
import libcst.matchers as m

from .base import BaseBug


class _BooleanFlipper(cst.CSTTransformer):
    def __init__(self, rng: random.Random, rate: float):
        super().__init__()
        self.rng = rng
        self.rate = rate
        self.sites = 0

    def leave_BooleanOperation(self, original_node, updated_node):
        operator = updated_node.operator
        if isinstance(operator, cst.And):
            new_cls = cst.Or
        elif isinstance(operator, cst.Or):
            new_cls = cst.And
        else:
            return updated_node
        if self.rng.random() >= self.rate:
            return updated_node
        self.sites += 1
        new_operator = new_cls(
            whitespace_before=operator.whitespace_before,
            whitespace_after=operator.whitespace_after,
        )
        return updated_node.with_changes(operator=new_operator)

    def leave_Comparison(self, original_node, updated_node):
        if len(updated_node.comparisons) != 1:
            return updated_node
        if m.findall(updated_node, m.Name("__name__")):
            return updated_node
        operator = updated_node.comparisons[0].operator
        if isinstance(operator, cst.Equal):
            new_cls = cst.NotEqual
        elif isinstance(operator, cst.NotEqual):
            new_cls = cst.Equal
        elif isinstance(operator, cst.In):
            new_cls = cst.NotIn
        elif isinstance(operator, cst.NotIn):
            new_cls = cst.In
        elif isinstance(operator, cst.Is):
            new_cls = cst.IsNot
        elif isinstance(operator, cst.IsNot):
            new_cls = cst.Is
        else:
            return updated_node
        if self.rng.random() >= self.rate:
            return updated_node
        self.sites += 1
        new_operator = new_cls(
            whitespace_before=operator.whitespace_before,
            whitespace_after=operator.whitespace_after,
        )
        return updated_node.with_changes(
            comparisons=[updated_node.comparisons[0].with_changes(operator=new_operator)]
        )

    def leave_UnaryOperation(self, original_node, updated_node):
        if not isinstance(updated_node.operator, cst.Not):
            return updated_node
        if self.rng.random() >= self.rate:
            return updated_node
        self.sites += 1
        inner = updated_node.expression
        if updated_node.lpar:
            # Μεταφορά τυχόν εξωτερικών παρενθέσεων του `(not x)` στο x.
            inner = inner.with_changes(
                lpar=[*updated_node.lpar, *inner.lpar],
                rpar=[*inner.rpar, *updated_node.rpar],
            )
        return inner


class BooleanLogicBug(BaseBug):
    smell_id = "boolean_logic"
    rule_id = "CWE-480"

    def apply(self, module, rng):
        flipper = _BooleanFlipper(rng, self.rate)
        new_module = module.visit(flipper)
        return new_module, flipper.sites
