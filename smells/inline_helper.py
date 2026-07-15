"""Smell: long methods μέσω inlining helpers — SonarQube rule S138.

Βρίσκει module-level συναρτήσεις που καλούνται με απλό τρόπο μέσα σε άλλες
συναρτήσεις και «κολλάει» το σώμα τους στο σημείο κλήσης:

    x = helper(a, b)   →   param1 = a
                           param2 = b
                           <σώμα helper>
                           x = <return expression>

Έτσι οι callers γίνονται μακριές, μονολιθικές συναρτήσεις. Ο ορισμός του
helper ΔΙΑΤΗΡΕΙΤΑΙ: άλλα αρχεία μπορεί να τον κάνουν import, και επιπλέον
μένει πίσω ως unused/dead code — bonus smell.

Συντηρητικές προϋποθέσεις (skip αν δεν πληρούνται — ποτέ ρίσκο):
- helper: module-level, χωρίς decorators/async/yield/await/global/nonlocal,
  χωρίς *args/**kwargs/keyword-only/positional-only, χωρίς αναδρομή,
  defaults μόνο literals (τα call-time vs def-time mutable defaults
  διαφέρουν σημασιολογικά), return μόνο ως τελευταίο statement (nested
  return θα «δραπέτευε» από τον caller), σώμα ≤ 25 statements
- call site: σκέτη κλήση ή απλή ανάθεση, μόνο positional args, μέσα σε
  συνάρτηση (όχι module level)
- ονόματα: όσα δένει ο helper δεν εμφανίζονται πουθενά στον caller, και
  τα ελεύθερα ονόματα του helper δεν σκιάζονται από locals του caller
"""
import builtins
import random

import libcst as cst
import libcst.matchers as m

from .base import BaseSmell, uses_introspection

_BUILTINS = frozenset(dir(builtins))
_LITERAL_NAMES = {"None", "True", "False"}
_MAX_HELPER_STATEMENTS = 25
_NO_RETURN = object()


def _is_literal_default(node):
    if node is None:
        return True
    if isinstance(node, (cst.Integer, cst.Float, cst.SimpleString)):
        return True
    if isinstance(node, cst.Name) and node.value in _LITERAL_NAMES:
        return True
    if isinstance(node, cst.UnaryOperation) and isinstance(
        node.expression, (cst.Integer, cst.Float)
    ):
        return True
    return False


class _NameSetCollector(cst.CSTVisitor):
    def __init__(self):
        super().__init__()
        self.names = set()

    def visit_Name(self, node):
        self.names.add(node.value)


class _BindingCollector(cst.CSTVisitor):
    """Ονόματα που γίνονται bind μέσα στο subtree (μεταβλητές, params,
    defs, imports, with/except aliases, comprehension targets)."""

    def __init__(self):
        super().__init__()
        self.bound = set()

    def _from_target(self, expr):
        if isinstance(expr, cst.Name):
            self.bound.add(expr.value)
        elif isinstance(expr, (cst.Tuple, cst.List)):
            for element in expr.elements:
                self._from_target(element.value)
        elif isinstance(expr, cst.StarredElement):
            self._from_target(expr.value)

    def visit_Assign(self, node):
        for target in node.targets:
            self._from_target(target.target)

    def visit_AnnAssign(self, node):
        self._from_target(node.target)

    def visit_AugAssign(self, node):
        self._from_target(node.target)

    def visit_For(self, node):
        self._from_target(node.target)

    def visit_NamedExpr(self, node):
        self._from_target(node.target)

    def visit_CompFor(self, node):
        self._from_target(node.target)

    def visit_Param(self, node):
        self.bound.add(node.name.value)

    def visit_FunctionDef(self, node):
        self.bound.add(node.name.value)

    def visit_ClassDef(self, node):
        self.bound.add(node.name.value)

    def visit_WithItem(self, node):
        if node.asname is not None:
            self._from_target(node.asname.name)

    def visit_ExceptHandler(self, node):
        if node.name is not None and isinstance(node.name.name, cst.Name):
            self.bound.add(node.name.name.value)

    def visit_Global(self, node):
        for item in node.names:
            self.bound.add(item.name.value)

    def visit_Nonlocal(self, node):
        for item in node.names:
            self.bound.add(item.name.value)

    def visit_ImportAlias(self, node):
        if node.asname is not None:
            self._from_target(node.asname.name)
        else:
            name = node.name
            while isinstance(name, cst.Attribute):
                name = name.value
            if isinstance(name, cst.Name):
                self.bound.add(name.value)


class _HelperInfo:
    __slots__ = ("params", "body_stmts", "return_expr", "bound", "free")

    def __init__(self, params, body_stmts, return_expr, bound, free):
        self.params = params          # list[(name, default_node_ή_None)]
        self.body_stmts = body_stmts  # σώμα χωρίς το τελικό return
        self.return_expr = return_expr  # expr, None (return σκέτο) ή _NO_RETURN
        self.bound = bound
        self.free = free


def _analyze_helper(fn):
    """Επιστρέφει _HelperInfo αν η συνάρτηση είναι ασφαλής για inlining."""
    if fn.decorators or fn.asynchronous is not None:
        return None
    params_obj = fn.params
    if (
        params_obj.posonly_params
        or params_obj.kwonly_params
        or isinstance(params_obj.star_arg, (cst.Param, cst.ParamStar))
        or params_obj.star_kwarg is not None
    ):
        return None
    if not isinstance(fn.body, cst.IndentedBlock):
        return None
    body = list(fn.body.body)
    if not body or len(body) > _MAX_HELPER_STATEMENTS:
        return None
    if m.findall(fn.body, m.Yield() | m.Await() | m.Global() | m.Nonlocal()):
        return None
    if uses_introspection(fn):
        return None
    if m.findall(fn.body, m.Name(fn.name.value)):
        return None  # αναδρομή/αυτοαναφορά

    params = []
    for param in params_obj.params:
        default = param.default
        if not _is_literal_default(default):
            return None
        params.append((param.name.value, default))

    returns = m.findall(fn.body, m.Return())
    if not returns:
        return_expr = _NO_RETURN
        body_stmts = body
    else:
        last = body[-1]
        if not (
            len(returns) == 1
            and isinstance(last, cst.SimpleStatementLine)
            and len(last.body) == 1
            and last.body[0] is returns[0]
        ):
            return None
        return_expr = returns[0].value  # μπορεί να είναι None (σκέτο return)
        body_stmts = body[:-1]

    all_names = _NameSetCollector()
    fn.visit(all_names)
    bindings = _BindingCollector()
    fn.visit(bindings)
    # Το όνομα του ίδιου του helper δεν είναι «τοπικό binding» — αν έμενε
    # στο σύνολο, θα συγκρουόταν πάντα με το call site στον caller.
    bound = bindings.bound - {fn.name.value}
    free = all_names.names - bound - _BUILTINS - {fn.name.value}
    return _HelperInfo(params, body_stmts, return_expr, bound, free)


class _Inliner(cst.CSTTransformer):
    def __init__(self, helpers, rng: random.Random, rate: float):
        super().__init__()
        self.helpers = helpers
        self.rng = rng
        self.rate = rate
        self.sites = 0
        self._stack = []

    def visit_FunctionDef(self, node):
        all_names = _NameSetCollector()
        node.visit(all_names)
        bindings = _BindingCollector()
        node.visit(bindings)
        self._stack.append(
            (all_names.names, bindings.bound, uses_introspection(node))
        )
        return True

    def leave_FunctionDef(self, original_node, updated_node):
        self._stack.pop()
        return updated_node

    def leave_SimpleStatementLine(self, original_node, updated_node):
        if not self._stack:
            return updated_node  # μόνο μέσα σε συναρτήσεις (long METHOD)
        caller_names, caller_bound, caller_introspects = self._stack[-1]
        if caller_introspects:
            return updated_node
        if len(updated_node.body) != 1:
            return updated_node

        stmt = updated_node.body[0]
        assign_targets = None
        if isinstance(stmt, cst.Expr) and isinstance(stmt.value, cst.Call):
            call = stmt.value
        elif (
            isinstance(stmt, cst.Assign)
            and isinstance(stmt.value, cst.Call)
            and len(stmt.targets) == 1
        ):
            call = stmt.value
            assign_targets = stmt.targets
        else:
            return updated_node

        if not isinstance(call.func, cst.Name):
            return updated_node
        info = self.helpers.get(call.func.value)
        if info is None:
            return updated_node
        for arg in call.args:
            if arg.keyword is not None or arg.star:
                return updated_node
        n_params = len(info.params)
        n_required = sum(1 for _, d in info.params if d is None)
        if not (n_required <= len(call.args) <= n_params):
            return updated_node
        # Έλεγχοι σύγκρουσης ονομάτων — βλ. module docstring.
        if info.bound & caller_names or info.free & caller_bound:
            return updated_node
        if self.rng.random() >= self.rate:
            return updated_node

        new_stmts = []
        for i, (pname, default) in enumerate(info.params):
            value = call.args[i].value if i < len(call.args) else default.deep_clone()
            new_stmts.append(
                cst.SimpleStatementLine(
                    body=[
                        cst.Assign(
                            targets=[cst.AssignTarget(target=cst.Name(pname))],
                            value=value,
                        )
                    ]
                )
            )
        for body_stmt in info.body_stmts:
            new_stmts.append(body_stmt.deep_clone())
        if assign_targets is not None:
            if info.return_expr is _NO_RETURN or info.return_expr is None:
                ret_value = cst.Name("None")
            else:
                ret_value = info.return_expr.deep_clone()
            new_stmts.append(
                cst.SimpleStatementLine(
                    body=[cst.Assign(targets=list(assign_targets), value=ret_value)]
                )
            )
        elif info.return_expr is not _NO_RETURN and info.return_expr is not None:
            # Σκέτη κλήση: το return expression μπορεί να έχει side effects,
            # οπότε διατηρείται ως expression statement.
            new_stmts.append(
                cst.SimpleStatementLine(body=[cst.Expr(value=info.return_expr.deep_clone())])
            )
        if not new_stmts:
            return updated_node

        new_stmts[0] = new_stmts[0].with_changes(
            leading_lines=updated_node.leading_lines
        )
        self.sites += 1
        return cst.FlattenSentinel(new_stmts)


class InlineHelperSmell(BaseSmell):
    smell_id = "inline_helper"
    rule_id = "S138"

    def apply(self, module, rng):
        helpers = {}
        seen = set()
        for stmt in module.body:
            if isinstance(stmt, cst.FunctionDef):
                name = stmt.name.value
                if name in seen:
                    # Διπλός ορισμός στο module → άγνωστο ποιος ισχύει, skip.
                    helpers.pop(name, None)
                    continue
                seen.add(name)
                info = _analyze_helper(stmt)
                if info is not None:
                    helpers[name] = info
        if not helpers:
            return module, 0
        inliner = _Inliner(helpers, rng, self.rate)
        new_module = module.visit(inliner)
        return new_module, inliner.sites
