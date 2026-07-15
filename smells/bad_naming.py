"""Smell: μη-περιγραφικά ονόματα locals — SonarQube rule S117.

Μετονομάζει local variables συναρτήσεων σε ονόματα τύπου x1/tmp/data2,
με scope-aware ανάλυση (libcst ScopeProvider) ώστε binding και όλες οι
αναφορές να μετονομάζονται μαζί, χωρίς να αγγίζονται ομώνυμες μεταβλητές
άλλων scopes ή attributes.

Συνειδητές εξαιρέσεις (για να ΜΗ σπάσει λειτουργικότητα):
- Παράμετροι ΔΕΝ μετονομάζονται: εξωτερικά call sites (και σε άλλα αρχεία
  του corpus) μπορεί να τις περνούν ως keyword arguments (f(alpha=1)) —
  το scope analysis ενός αρχείου δεν το βλέπει αυτό.
- self/cls/_ και dunders εξαιρούνται.
- Συναρτήσεις που καλούν eval/exec/locals/globals/vars εξαιρούνται
  ολόκληρες: τα ονόματα εκεί μπορεί να χρησιμοποιούνται δυναμικά.
- Bindings σε μη-υποστηριζόμενες μορφές (with/except/comprehensions,
  nested def) → skip της μεταβλητής, όχι ρίσκο.
"""
import builtins
import keyword
import random
import re

import libcst as cst
import libcst.matchers as m
from libcst.metadata import (
    Assignment,
    FunctionScope,
    MetadataWrapper,
    PositionProvider,
    ScopeProvider,
)

from .base import BaseSmell, uses_introspection

_BAD_BASES = [
    "x", "y", "z", "a", "b", "aa", "zz", "q",
    "tmp", "temp", "data", "var", "val", "v",
    "obj", "res", "thing", "stuff", "foo", "my_var",
]
_PROTECTED = {"self", "cls", "_"}
_BUILTINS = frozenset(dir(builtins))
_DUNDER_RE = re.compile(r"^__.*__$")


class _AllNamesCollector(cst.CSTVisitor):
    def __init__(self):
        super().__init__()
        self.names = set()

    def visit_Name(self, node):
        self.names.add(node.value)


class _Renamer(cst.CSTTransformer):
    """Μετονομάζει συγκεκριμένα Name nodes με ταυτότητα αντικειμένου."""

    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping

    def leave_Name(self, original_node, updated_node):
        new_name = self.mapping.get(original_node)
        if new_name is None:
            return updated_node
        return updated_node.with_changes(value=new_name)


def _declared_names(node):
    """Ονόματα σε nonlocal/global δηλώσεις μέσα στο subtree του scope.

    Αν μετονομαστεί μεταβλητή που nested function δηλώνει ως nonlocal, το
    ίδιο το statement `nonlocal x` δεν είναι ούτε binding ούτε access στο
    scope analysis και θα έμενε αμετονόμαστο → SyntaxError. Skip.
    """
    names = set()
    for decl in m.findall(node, m.Nonlocal()):
        for item in decl.names:
            names.add(item.name.value)
    for decl in m.findall(node, m.Global()):
        for item in decl.names:
            names.add(item.name.value)
    return names


def _collect_binding_targets(expr, name, out):
    """Μαζεύει ΜΟΝΟ τα Name nodes που κάνουν binding μέσα σε assignment
    target. Attributes/Subscripts (obj.x, arr[i]) δεν κάνουν bind ονόματος
    και αγνοούνται — ένα σκέτο findall θα έπιανε λάθος και το obj.x."""
    if isinstance(expr, cst.Name):
        if expr.value == name:
            out.append(expr)
    elif isinstance(expr, (cst.Tuple, cst.List)):
        for element in expr.elements:
            _collect_binding_targets(element.value, name, out)
    elif isinstance(expr, cst.StarredElement):
        _collect_binding_targets(expr.value, name, out)


def _binding_name_nodes(assignment_node, name):
    """Τα Name nodes του binding· None αν το είδος δεν υποστηρίζεται
    (συντηρητικό: ο caller κάνει skip τη μεταβλητή)."""
    if isinstance(assignment_node, cst.Name):
        return [assignment_node] if assignment_node.value == name else None
    if isinstance(assignment_node, cst.Assign):
        targets = [t.target for t in assignment_node.targets]
    elif isinstance(assignment_node, (cst.AnnAssign, cst.AugAssign)):
        targets = [assignment_node.target]
    elif isinstance(assignment_node, cst.For):
        targets = [assignment_node.target]
    elif isinstance(assignment_node, cst.NamedExpr):
        targets = [assignment_node.target]
    else:
        return None
    found = []
    for target in targets:
        _collect_binding_targets(target, name, found)
    return found or None


def _new_bad_name(rng, used_names):
    for _ in range(200):
        base = rng.choice(_BAD_BASES)
        candidate = base if rng.random() < 0.3 else f"{base}{rng.randint(1, 99)}"
        if candidate not in used_names and not keyword.iskeyword(candidate):
            return candidate
    counter = 0
    while f"v{counter}" in used_names:
        counter += 1
    return f"v{counter}"


class BadNamingSmell(BaseSmell):
    smell_id = "bad_naming"
    rule_id = "S117"

    def apply(self, module, rng):
        collector = _AllNamesCollector()
        module.visit(collector)
        # Τα νέα ονόματα αποφεύγουν ΚΑΘΕ identifier του module (συν builtins
        # και keywords) ώστε να αποκλείεται οποιοδήποτε shadowing.
        used_names = set(collector.names) | _BUILTINS | set(keyword.kwlist)

        wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
        scope_map = wrapper.resolve(ScopeProvider)
        positions = wrapper.resolve(PositionProvider)

        func_scopes = {
            s for s in scope_map.values() if isinstance(s, FunctionScope)
        }

        # Ταξινόμηση scopes κατά θέση στο source: τα sets έχουν
        # μη-ντετερμινιστική σειρά iteration (id-based hashing) και θα
        # χαλούσαν το reproducibility του seed μεταξύ runs.
        def _source_position(scope):
            pos = positions.get(scope.node)
            return (pos.start.line, pos.start.column) if pos else (0, 0)

        mapping = {}
        sites = 0
        for scope in sorted(func_scopes, key=_source_position):
            if uses_introspection(scope.node):
                continue
            declared = _declared_names(scope.node)
            by_name = {}
            for asgn in scope.assignments:
                by_name.setdefault(asgn.name, []).append(asgn)
            for name in sorted(by_name):
                asgns = by_name[name]
                if name in _PROTECTED or _DUNDER_RE.match(name) or name in declared:
                    continue
                if not all(isinstance(a, Assignment) for a in asgns):
                    continue
                if any(isinstance(a.node, cst.Param) for a in asgns):
                    continue
                binding_nodes = []
                supported = True
                for asgn in asgns:
                    nodes = _binding_name_nodes(asgn.node, name)
                    if nodes is None:
                        supported = False
                        break
                    binding_nodes.extend(nodes)
                if not supported:
                    continue
                if rng.random() >= self.rate:
                    continue
                new_name = _new_bad_name(rng, used_names)
                used_names.add(new_name)
                for node in binding_nodes:
                    mapping[node] = new_name
                for asgn in asgns:
                    for access in asgn.references:
                        if isinstance(access.node, cst.Name):
                            mapping[access.node] = new_name
                sites += 1

        if not mapping:
            return module, 0
        return module.visit(_Renamer(mapping)), sites
