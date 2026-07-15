"""Smell: μακριά λίστα παραμέτρων — SonarQube rule S107.

Προσθέτει 2-5 αχρησιμοποίητες παραμέτρους σε ορισμούς συναρτήσεων. Όλες
παίρνουν default τιμή, οπότε ΚΑΝΕΝΑ υπάρχον call site δεν σπάει — ούτε σε
άλλα αρχεία του corpus. Το signature όμως γίνεται φορτωμένο/παραπλανητικό.

Εξαιρέσεις για ασφάλεια:
- Συναρτήσεις με decorators: frameworks (click, pytest fixtures κτλ.)
  κάνουν introspection στο signature και θα άλλαζαν συμπεριφορά.
- Ονόματα test*: το pytest ερμηνεύει κάθε παράμετρο test function ως
  fixture request → θα έσπαγε το collection.
- Συναρτήσεις με **kwargs: ένας caller μπορεί να περνάει ήδη keyword με
  το ίδιο όνομα ενός junk param, που θα άλλαζε πού καταλήγει η τιμή.
- Dunders και συναρτήσεις με eval/exec/locals/globals/vars.
"""
import random

import libcst as cst

from .base import BaseSmell, uses_introspection

_JUNK_PARAMS = [
    ("options", "None"),
    ("verbose", "False"),
    ("ctx", "None"),
    ("timeout", "None"),
    ("debug", "False"),
    ("strict", "False"),
    ("callback", "None"),
    ("metadata", "None"),
    ("flags", "0"),
    ("mode", '"default"'),
    ("config", "None"),
    ("cache_enabled", "True"),
    ("retries", "0"),
    ("logger", "None"),
    ("extra", "None"),
]


class _AllNamesCollector(cst.CSTVisitor):
    def __init__(self):
        super().__init__()
        self.names = set()

    def visit_Name(self, node):
        self.names.add(node.value)


class _ParamStuffer(cst.CSTTransformer):
    def __init__(self, rng: random.Random, rate: float, used_names: set):
        super().__init__()
        self.rng = rng
        self.rate = rate
        self.used_names = used_names
        self.sites = 0

    def leave_FunctionDef(self, original_node, updated_node):
        if updated_node.decorators:
            return updated_node
        name = updated_node.name.value
        if name.startswith("test") or (name.startswith("__") and name.endswith("__")):
            return updated_node
        params = updated_node.params
        if params.star_kwarg is not None:
            return updated_node
        if uses_introspection(updated_node):
            return updated_node
        if self.rng.random() >= self.rate:
            return updated_node
        # Junk ονόματα που δεν συγκρούονται με ΤΙΠΟΤΑ στο module — αν η
        # συνάρτηση διάβαζε global `logger` και προσθέταμε param `logger`,
        # το body θα διάβαζε ξαφνικά το param (σκιά) → αλλαγή συμπεριφοράς.
        pool = [(n, d) for n, d in _JUNK_PARAMS if n not in self.used_names]
        if len(pool) < 2:
            return updated_node
        count = min(self.rng.randint(2, 5), len(pool))
        chosen = self.rng.sample(pool, count)
        new_params = [
            cst.Param(name=cst.Name(n), default=cst.parse_expression(d))
            for n, d in chosen
        ]
        self.sites += 1
        return updated_node.with_changes(
            params=params.with_changes(params=[*params.params, *new_params])
        )


class LongParameterListSmell(BaseSmell):
    smell_id = "long_parameter_list"
    rule_id = "S107"

    def apply(self, module, rng):
        collector = _AllNamesCollector()
        module.visit(collector)
        stuffer = _ParamStuffer(rng, self.rate, collector.names)
        new_module = module.visit(stuffer)
        return new_module, stuffer.sites
