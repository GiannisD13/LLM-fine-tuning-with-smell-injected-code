"""Smell: νεκρός κώδικας — SonarQube rules S125/S1481/S2583.

Τρεις παραλλαγές, όλες χωρίς καμία επίδραση στη συμπεριφορά:
- unused_var: αχρησιμοποίητη τοπική μεταβλητή με literal τιμή (S1481)
- if_false: `if False:` block με αντίγραφο πραγματικού statement του ίδιου
  σώματος — μοιάζει αυθεντικό, δεν εκτελείται ποτέ (S2583)
- commented: πραγματικές γραμμές του σώματος ως σχόλια — commented-out
  code (S125). Εδώ αξιοποιείται ότι το libcst χειρίζεται comments.

Δεν αντιγράφονται statements με global/nonlocal: αυτές οι δηλώσεις
ισχύουν για ΟΛΟ το scope σε compile time, ακόμα και μέσα σε `if False:`.
"""
import random

import libcst as cst
import libcst.matchers as m

from .base import BaseSmell, uses_introspection

_VAR_NAMES = [
    "unused_result", "temp_data", "old_value", "backup_state",
    "cached_result", "legacy_data", "tmp_holder", "prev_result",
]
_JUNK_VALUES = ["None", "0", "-1", "[]", "{}", '""', "True"]
_MAX_SOURCE_CHARS = 300
_MAX_COMMENT_LINES = 5


def _is_docstring(stmt):
    return (
        isinstance(stmt, cst.SimpleStatementLine)
        and len(stmt.body) == 1
        and isinstance(stmt.body[0], cst.Expr)
        and isinstance(stmt.body[0].value, (cst.SimpleString, cst.ConcatenatedString))
    )


class _DeadCodeInjector(cst.CSTTransformer):
    def __init__(self, rng: random.Random, rate: float, module: cst.Module, used_names: set):
        super().__init__()
        self.rng = rng
        self.rate = rate
        self.module = module
        self.used_names = used_names
        self.sites = 0

    def _fresh_name(self):
        base = self.rng.choice(_VAR_NAMES)
        candidate = base
        counter = 2
        while candidate in self.used_names:
            candidate = f"{base}{counter}"
            counter += 1
        self.used_names.add(candidate)
        return candidate

    def _pick_source(self, stmts):
        candidates = []
        for stmt in stmts:
            if m.findall(stmt, m.Global() | m.Nonlocal()):
                continue
            if len(self.module.code_for_node(stmt)) > _MAX_SOURCE_CHARS:
                continue
            candidates.append(stmt)
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def leave_FunctionDef(self, original_node, updated_node):
        # locals()/vars() θα «έβλεπαν» τις junk μεταβλητές → αλλαγή
        # συμπεριφοράς, οπότε skip όλη τη συνάρτηση.
        if uses_introspection(updated_node):
            return updated_node
        body = updated_node.body
        if not isinstance(body, cst.IndentedBlock) or not body.body:
            return updated_node
        if self.rng.random() >= self.rate:
            return updated_node

        stmts = list(body.body)
        original_stmts = list(body.body)
        start = 1 if _is_docstring(stmts[0]) else 0
        variants = self.rng.sample(
            ["unused_var", "if_false", "commented"], self.rng.randint(1, 2)
        )
        injected = 0
        for variant in variants:
            if variant == "unused_var":
                line = cst.SimpleStatementLine(
                    body=[
                        cst.Assign(
                            targets=[cst.AssignTarget(target=cst.Name(self._fresh_name()))],
                            value=cst.parse_expression(self.rng.choice(_JUNK_VALUES)),
                        )
                    ]
                )
                stmts.insert(self.rng.randint(start, len(stmts)), line)
                injected += 1
            elif variant == "if_false":
                source = self._pick_source(original_stmts[start:])
                if source is not None:
                    inner = source.deep_clone().with_changes(leading_lines=[])
                else:
                    inner = cst.SimpleStatementLine(body=[cst.Pass()])
                block = cst.If(
                    test=cst.Name("False"), body=cst.IndentedBlock(body=[inner])
                )
                stmts.insert(self.rng.randint(start, len(stmts)), block)
                injected += 1
            else:
                if len(original_stmts) <= start:
                    continue
                source = self.rng.choice(original_stmts[start:])
                text = self.module.code_for_node(source)
                lines = [ln for ln in text.splitlines() if ln.strip()][:_MAX_COMMENT_LINES]
                if not lines:
                    continue
                comments = [
                    cst.EmptyLine(indent=True, comment=cst.Comment("# " + ln.rstrip()))
                    for ln in lines
                ]
                idx = self.rng.randint(start, len(stmts) - 1) if len(stmts) > start else start
                stmts[idx] = stmts[idx].with_changes(
                    leading_lines=[*stmts[idx].leading_lines, *comments]
                )
                injected += 1

        if injected == 0:
            return updated_node
        self.sites += injected
        return updated_node.with_changes(body=body.with_changes(body=stmts))


class DeadCodeSmell(BaseSmell):
    smell_id = "dead_code"
    rule_id = "S125"

    def apply(self, module, rng):
        collector_names = set()

        class _Names(cst.CSTVisitor):
            def visit_Name(self, node):
                collector_names.add(node.value)

        module.visit(_Names())
        injector = _DeadCodeInjector(rng, self.rate, module, collector_names)
        new_module = module.visit(injector)
        return new_module, injector.sites
