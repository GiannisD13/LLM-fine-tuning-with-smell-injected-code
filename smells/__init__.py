"""Registry των διαθέσιμων smell transformers (semantic-preserving)."""
from .bad_naming import BadNamingSmell
from .dead_code import DeadCodeSmell
from .deep_nesting import DeepNestingSmell
from .inline_helper import InlineHelperSmell
from .long_parameter_list import LongParameterListSmell
from .magic_numbers import MagicNumberSmell
from .missing_docstrings import MissingDocstringSmell

ALL_SMELLS = [
    BadNamingSmell,
    MagicNumberSmell,
    DeepNestingSmell,
    MissingDocstringSmell,
    LongParameterListSmell,
    DeadCodeSmell,
    InlineHelperSmell,
]

SMELL_REGISTRY = {cls.smell_id: cls for cls in ALL_SMELLS}
