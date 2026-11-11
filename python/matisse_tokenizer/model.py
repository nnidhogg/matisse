"""The encoding a model uses, by its name, as tiktoken's model module gives it: encoding_name_for_model() and
encoding_for_model() over tiktoken 0.14.0's tables, MODEL_TO_ENCODING and MODEL_PREFIX_TO_ENCODING, restricted to the
encodings the package serves.

A model tiktoken maps to an encoding the package does not serve, p50k_base or p50k_edit, is refused with KeyError
naming that encoding, and a model tiktoken does not know is refused with tiktoken's KeyError.
"""

from __future__ import annotations

from matisse_tokenizer import core, registry

# The tables of tiktoken 0.14.0's tiktoken/model.py, in its order, with the entries whose encodings the package serves.
MODEL_PREFIX_TO_ENCODING: dict[str, str] = {
    "o1-": "o200k_base",
    "o3-": "o200k_base",
    "o4-mini-": "o200k_base",
    # chat
    "gpt-5": "o200k_base",
    "gpt-4.5-": "o200k_base",
    "gpt-4.1-": "o200k_base",
    "chatgpt-4o-": "o200k_base",
    "gpt-4o-": "o200k_base",
    "gpt-4-": "cl100k_base",
    "gpt-3.5-turbo-": "cl100k_base",
    "gpt-35-turbo-": "cl100k_base",
    "gpt-oss-": "o200k_harmony",
    # fine-tuned
    "ft:gpt-4o": "o200k_base",
    "ft:gpt-4": "cl100k_base",
    "ft:gpt-3.5-turbo": "cl100k_base",
    "ft:davinci-002": "cl100k_base",
    "ft:babbage-002": "cl100k_base",
}
"""The encoding of a model whose name begins with a prefix, the first prefix that matches deciding."""

MODEL_TO_ENCODING: dict[str, str] = {
    # reasoning
    "o1": "o200k_base",
    "o3": "o200k_base",
    "o4-mini": "o200k_base",
    # chat
    "gpt-5": "o200k_base",
    "gpt-4.1": "o200k_base",
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "gpt-35-turbo": "cl100k_base",
    # base
    "davinci-002": "cl100k_base",
    "babbage-002": "cl100k_base",
    # embeddings
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # text (deprecated)
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
    # old embeddings (deprecated)
    "text-similarity-davinci-001": "r50k_base",
    "text-similarity-curie-001": "r50k_base",
    "text-similarity-babbage-001": "r50k_base",
    "text-similarity-ada-001": "r50k_base",
    "text-search-davinci-doc-001": "r50k_base",
    "text-search-curie-doc-001": "r50k_base",
    "text-search-babbage-doc-001": "r50k_base",
    "text-search-ada-doc-001": "r50k_base",
    "code-search-babbage-code-001": "r50k_base",
    "code-search-ada-code-001": "r50k_base",
    # open source
    "gpt2": "gpt2",
    "gpt-2": "gpt2",
}
"""The encoding of a model by its exact name, looked up before the prefixes."""

UNSERVED_MODEL_TO_ENCODING: dict[str, str] = {
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "code-davinci-002": "p50k_base",
    "code-davinci-001": "p50k_base",
    "code-cushman-002": "p50k_base",
    "code-cushman-001": "p50k_base",
    "davinci-codex": "p50k_base",
    "cushman-codex": "p50k_base",
    "text-davinci-edit-001": "p50k_edit",
    "code-davinci-edit-001": "p50k_edit",
}
"""The models of tiktoken 0.14.0's table whose encodings the package does not serve, which are refused naming them."""


def encoding_name_for_model(model_name: str) -> str:
    """Returns the name of the encoding a model uses, by its exact name and then by the first prefix it begins with.

    Args:
        model_name: The model's name.

    Returns:
        The encoding's name.

    Raises:
        KeyError: If tiktoken maps the model to an encoding the package does not serve, naming it, or knows no
            encoding for it, with tiktoken's message.
    """
    if model_name in MODEL_TO_ENCODING:
        return MODEL_TO_ENCODING[model_name]
    if model_name in UNSERVED_MODEL_TO_ENCODING:
        raise KeyError(
            f"{model_name} uses the encoding {UNSERVED_MODEL_TO_ENCODING[model_name]}, which matisse_tokenizer does "
            f"not serve."
        )
    for model_prefix, encoding_name in MODEL_PREFIX_TO_ENCODING.items():
        if model_name.startswith(model_prefix):
            return encoding_name
    raise KeyError(
        f"Could not automatically map {model_name} to a tokeniser. "
        "Please use `tiktoken.get_encoding` to explicitly get the tokeniser you expect."
    ) from None


def encoding_for_model(model_name: str) -> core.Encoding:
    """Returns the encoding a model uses, as get_encoding() gives it.

    Args:
        model_name: The model's name.

    Returns:
        The encoding.

    Raises:
        KeyError: As encoding_name_for_model() refuses the model.
    """
    return registry.get_encoding(encoding_name_for_model(model_name))
