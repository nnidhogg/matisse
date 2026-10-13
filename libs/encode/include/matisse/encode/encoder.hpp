#ifndef MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_ENCODER_HPP
#define MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_ENCODER_HPP

/**
 * @file
 * @brief The encoder, Encoder with encode(), chunk_boundaries(), encode_parallel() and decode(): the token ids of a
 *        text under a definition, read sequentially or in parallel by cutting only at certified anchors, the ids
 *        identical either way.
 *
 * encode() splits the text at every allowed special token, segments each stretch between them as a complete input, and
 * merges every piece alone by the definition's merge rule, encoder.py's pair loop or tiktoken's rank merge.
 * chunk_boundaries() walks forward from each equal byte target to the first anchor of the inventory's window table, and
 * encode_parallel() gives each chunk to a worker that segments it with the decided right context and keeps the pieces
 * ending at its cut. The encoder claims one thing: for every text the definition accepts, encode_parallel() returns the
 * ids encode() returns, at every chunk count. Nothing is declared here yet; docs/design.md sketches the declarations.
 */
namespace matisse::encode
{
} // namespace matisse::encode

#endif // MATISSE_LIBS_ENCODE_INCLUDE_MATISSE_ENCODE_ENCODER_HPP
