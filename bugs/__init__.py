"""Registry των διαθέσιμων bug transformers (semantic-breaking)."""
from .boolean_logic import BooleanLogicBug
from .off_by_one import OffByOneBug

ALL_BUGS = [
    OffByOneBug,
    BooleanLogicBug,
]

BUG_REGISTRY = {cls.smell_id: cls for cls in ALL_BUGS}
