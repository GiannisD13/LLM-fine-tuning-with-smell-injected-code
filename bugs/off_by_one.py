"""Bug: off-by-one errors — CWE-193.

Τρεις μορφές, όλες κλασικά ανθρώπινα λάθη ορίων:
- range(n) → range(n ± 1) (και το stop όρισμα σε range(a, b))
- < ↔ <= και > ↔ >= σε συγκρίσεις που περιέχουν αριθμό ή len(...)
  (περιορισμός πιθανοφάνειας: εκεί συμβαίνουν τα αληθινά boundary bugs)
- μετατόπιση ορίων σε slices: x[:n] → x[:n+1], x[i:] → x[i-1:]

Τα ± 1 μπαίνουν μόνο σε «απλά» expressions (Name/Integer/Attribute/
Subscript/Call): το libcst δεν προσθέτει παρενθέσεις αυτόματα, οπότε
wrapping σύνθετης έκφρασης θα άλλαζε προτεραιότητες με ανεξέλεγκτο τρόπο.
"""
import random

import libcst as cst
import libcst.matchers as m

from .base import BaseBug

_SIMPLE_NODES = (cst.Name, cst.Integer, cst.Attribute, cst.Subscript, cst.Call)

_ORDER_SWAPS = {
    cst.LessThan: cst.LessThanEqual,
    cst.LessThanEqual: cst.LessThan,
    cst.GreaterThan: cst.GreaterThanEqual,
    cst.GreaterThanEqual: cst.GreaterThan,
}


class _OffByOneInjector(cst.CSTTransformer):
    def __init__(self, rng: random.Random, rate: float):
        super().__init__()
        self.rng = rng
        self.rate = rate
        self.sites = 0

    def _bumpable(self, expr):
        return isinstance(expr, _SIMPLE_NODES)

    def _bump(self, expr):
        # Δεκαδικό literal: αλλάζει το ίδιο το νούμερο (10 → 11), όπως θα
        # έκανε ένας άνθρωπος — όχι «10 + 1» που φωνάζει ότι είναι τεχνητό.
        if isinstance(expr, cst.Integer) and expr.value.isdigit():
            delta = self.rng.choice([1, -1])
            new_value = int(expr.value) + delta
            if new_value < 0:
                new_value = int(expr.value) + 1
            return expr.with_changes(value=str(new_value))
        if isinstance(expr, _SIMPLE_NODES):
            operator = self.rng.choice([cst.Add(), cst.Subtract()])
            return cst.BinaryOperation(
                left=expr, operator=operator, right=cst.Integer("1")
            )
        return None

    def leave_Call(self, original_node, updated_node):
        func = updated_node.func
        if not (isinstance(func, cst.Name) and func.value == "range"):
            return updated_node
        args = updated_node.args
        if not args or len(args) > 3 or any(a.star or a.keyword for a in args):
            return updated_node
        stop_index = 0 if len(args) == 1 else 1
        if not self._bumpable(args[stop_index].value):
            return updated_node
        if self.rng.random() >= self.rate:
            return updated_node
        bumped = self._bump(args[stop_index].value)
        if bumped is None:
            return updated_node
        self.sites += 1
        new_args = list(args)
        new_args[stop_index] = args[stop_index].with_changes(value=bumped)
        return updated_node.with_changes(args=new_args)

    def leave_Comparison(self, original_node, updated_node):
        if len(updated_node.comparisons) != 1:
            return updated_node
        operator = updated_node.comparisons[0].operator
        swap_cls = _ORDER_SWAPS.get(type(operator))
        if swap_cls is None:
            return updated_node
        if not (
            m.findall(updated_node, m.Integer())
            or m.findall(updated_node, m.Call(func=m.Name("len")))
        ):
            return updated_node
        if self.rng.random() >= self.rate:
            return updated_node
        self.sites += 1
        new_operator = swap_cls(
            whitespace_before=operator.whitespace_before,
            whitespace_after=operator.whitespace_after,
        )
        return updated_node.with_changes(
            comparisons=[updated_node.comparisons[0].with_changes(operator=new_operator)]
        )

    def leave_Slice(self, original_node, updated_node):
        fields = []
        if updated_node.upper is not None and self._bumpable(updated_node.upper):
            fields.append("upper")
        if updated_node.lower is not None and self._bumpable(updated_node.lower):
            fields.append("lower")
        if not fields:
            return updated_node
        if self.rng.random() >= self.rate:
            return updated_node
        field = self.rng.choice(fields)
        bumped = self._bump(getattr(updated_node, field))
        if bumped is None:
            return updated_node
        self.sites += 1
        return updated_node.with_changes(**{field: bumped})


class OffByOneBug(BaseBug):
    smell_id = "off_by_one"
    rule_id = "CWE-193"

    def apply(self, module, rng):
        injector = _OffByOneInjector(rng, self.rate)
        new_module = module.visit(injector)
        return new_module, injector.sites
