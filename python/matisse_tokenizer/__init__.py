"""matisse-tokenizer: matisse's exact BPE encoding for GPT-2, cl100k_base, o200k_base and Llama 3, behind tiktoken's
interface, so that `import matisse_tokenizer as tiktoken` serves the common calls with tiktoken's ids.

get_encoding() fetches and caches a pinned rank file as tiktoken does, and load_encoding() reads one from a path, which
Llama 3's needs. docs/python.md in matisse's repository lists every call and where the package refuses or differs.
"""

from matisse_tokenizer.core import Encoding
from matisse_tokenizer.registry import get_encoding, list_encoding_names, load_encoding

__version__ = "0.1.0"

__all__ = [
    "Encoding",
    "encoding_for_model",
    "encoding_name_for_model",
    "get_encoding",
    "list_encoding_names",
    "load_encoding",
]


def encoding_name_for_model(model_name: str) -> str:
    """Refuses: matisse_tokenizer keeps no table of model names, which change with every model a vendor releases.

    Args:
        model_name: The model's name.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        f"matisse_tokenizer keeps no table of model names; call get_encoding() with the encoding {model_name!r} uses"
    )


def encoding_for_model(model_name: str) -> Encoding:
    """Refuses, as encoding_name_for_model() does.

    Args:
        model_name: The model's name.

    Raises:
        NotImplementedError: Always.
    """
    encoding_name_for_model(model_name)
