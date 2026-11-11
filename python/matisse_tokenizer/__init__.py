"""matisse-tokenizer: matisse's exact BPE encoding for GPT-2, cl100k_base, o200k_base and Llama 3, behind tiktoken's
interface, so that `import matisse_tokenizer as tiktoken` serves the common calls with tiktoken's ids.

get_encoding() fetches and caches a pinned rank file as tiktoken does, load_encoding() reads one from a path, which
Llama 3's needs, and load_tokenizer_json() reads a Hugging Face tokenizer.json under the pattern it carries.
docs/python.md in matisse's repository lists every call and where the package refuses or differs.
"""

# The package's one version: pyproject.toml reads it for the package's metadata, and load.py for the User-Agent, so it
# stands before the imports.
__version__ = "0.1.0"

from matisse_tokenizer.core import Encoding  # noqa: E402
from matisse_tokenizer.model import encoding_for_model, encoding_name_for_model  # noqa: E402
from matisse_tokenizer.registry import (  # noqa: E402
    get_encoding,
    list_encoding_names,
    load_encoding,
    load_tokenizer_json,
)

__all__ = [
    "Encoding",
    "encoding_for_model",
    "encoding_name_for_model",
    "get_encoding",
    "list_encoding_names",
    "load_encoding",
    "load_tokenizer_json",
]

