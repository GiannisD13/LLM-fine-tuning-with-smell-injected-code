"""Βάση για bug transformers.

Σε αντίθεση με τα smells, τα bugs ΑΛΛΑΖΟΥΝ σκόπιμα τη συμπεριφορά του
κώδικα — αυτή είναι η ουσία τους. Το μόνο invariant που διατηρείται είναι
η συντακτική εγκυρότητα (το compile() του pipeline περνάει πάντα).
"""
from smells.base import BaseSmell


class BaseBug(BaseSmell):
    kind = "bug"
