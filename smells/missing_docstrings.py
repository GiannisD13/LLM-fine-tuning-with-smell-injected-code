"""Smell: απόντα ή άχρηστα docstrings — SonarQube rule S1720.

Για κάθε docstring (module/class/function), με πιθανότητα rate είτε το
αφαιρεί εντελώς είτε το αντικαθιστά με γενικό, μη-πληροφοριακό docstring.
Αν το docstring είναι το μοναδικό statement σε function/class, μπαίνει
`pass` ώστε το σώμα να μη μείνει κενό (SyntaxError).
"""
import random

import libcst as cst

from .base import BaseSmell

_USELESS_DOCSTRINGS = [
    '"""TODO"""',
    '"""..."""',
    '"""Do stuff."""',
    '"""This is a function."""',
    '"""helper"""',
]


class _DocstringDegrader(cst.CSTTransformer):
    def __init__(self, rng: random.Random, rate: float):
        super().__init__()
        self.rng = rng
        self.rate = rate
        self.sites = 0

    def _degrade_block(self, stmts, needs_pass):
        """Επιστρέφει νέα λίστα statements ή None αν δεν υπάρχει docstring
        ή δεν επιλέχθηκε από το rate."""
        if not stmts:
            return None
        first = stmts[0]
        is_docstring = (
            isinstance(first, cst.SimpleStatementLine)
            and len(first.body) == 1
            and isinstance(first.body[0], cst.Expr)
            and isinstance(first.body[0].value, (cst.SimpleString, cst.ConcatenatedString))
        )
        if not is_docstring:
            return None
        if self.rng.random() >= self.rate:
            return None
        self.sites += 1
        rest = list(stmts[1:])
        if self.rng.random() < 0.5:
            if rest or not needs_pass:
                return rest
            return [cst.SimpleStatementLine(body=[cst.Pass()])]
        useless = cst.SimpleStatementLine(
            body=[cst.Expr(cst.SimpleString(self.rng.choice(_USELESS_DOCSTRINGS)))]
        )
        return [useless] + rest

    def _process_def(self, node):
        if not isinstance(node.body, cst.IndentedBlock):
            return node
        new_stmts = self._degrade_block(list(node.body.body), needs_pass=True)
        if new_stmts is None:
            return node
        return node.with_changes(body=node.body.with_changes(body=new_stmts))

    def leave_FunctionDef(self, original_node, updated_node):
        return self._process_def(updated_node)

    def leave_ClassDef(self, original_node, updated_node):
        return self._process_def(updated_node)

    def leave_Module(self, original_node, updated_node):
        # Το module docstring δεν χρειάζεται pass: κενό module είναι έγκυρο.
        new_stmts = self._degrade_block(list(updated_node.body), needs_pass=False)
        if new_stmts is None:
            return updated_node
        return updated_node.with_changes(body=new_stmts)


class MissingDocstringSmell(BaseSmell):
    smell_id = "missing_docstrings"
    rule_id = "S1720"

    def apply(self, module, rng):
        degrader = _DocstringDegrader(rng, self.rate)
        new_module = module.visit(degrader)
        return new_module, degrader.sites
