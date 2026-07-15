"""Κοινή βάση για όλους τους transformers (smells ΚΑΙ bugs).

Κάθε transformer δηλώνει smell_id (μπαίνει στο smells_applied/bugs_applied
του output JSONL), rule_id (SonarQube RSPEC ή CWE, για το paper) και kind
("smell" = διατηρεί τη συμπεριφορά, "bug" = την αλλάζει σκόπιμα).
"""
import random

import libcst as cst
import libcst.matchers as m

# Συναρτήσεις που κοιτάζουν δυναμικά τα τοπικά ονόματα: οποιαδήποτε
# προσθήκη/μετονομασία μεταβλητών εκεί μπορεί να αλλάξει συμπεριφορά.
_INTROSPECTION = frozenset({"eval", "exec", "locals", "globals", "vars"})


def uses_introspection(node):
    for call in m.findall(node, m.Call(func=m.Name())):
        if call.func.value in _INTROSPECTION:
            return True
    return False


class BaseSmell:
    smell_id: str = ""
    rule_id: str = ""
    kind: str = "smell"

    def __init__(self, rate: float):
        self.rate = rate

    def apply(self, module: cst.Module, rng: random.Random) -> tuple[cst.Module, int]:
        """Επιστρέφει (νέο module, πλήθος σημείων που αλλοιώθηκαν).

        sites == 0 σημαίνει ότι δεν βρέθηκε/επιλέχθηκε κατάλληλο σημείο —
        ο caller δεν προσμετρά τον transformer στο αντίστοιχο applied πεδίο.
        """
        raise NotImplementedError
