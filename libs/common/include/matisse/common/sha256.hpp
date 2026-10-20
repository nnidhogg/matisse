#ifndef MATISSE_LIBS_COMMON_INCLUDE_MATISSE_COMMON_SHA256_HPP
#define MATISSE_LIBS_COMMON_INCLUDE_MATISSE_COMMON_SHA256_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>

/**
 * @file
 * @brief SHA-256, sha256(), the digest the tests hold the pinned vocabularies and the generated class tables to, and
 *        the probes the corpora and their ids, in its three steps: the padding, one block's message schedule and its
 *        compression.
 *
 * The tests and the probes share it through matisse_common, a target in the build tree only, neither installed nor
 * exported, so that checking a digest needs no hashing library.
 */
namespace matisse::common
{
/**
 * @brief Returns a message padded as SHA-256 pads it: a one bit, zeros to 56 mod 64, and the bit length big-endian.
 * @param text The message.
 * @return The padded message, a whole number of 64-byte blocks.
 */
inline std::string padded_message(const std::string_view text)
{
    std::string padded{text};

    padded.push_back(static_cast<char>(0x80));

    while (padded.size() % 64 != 56)
    {
        padded.push_back('\0');
    }

    const auto bit_length{static_cast<std::uint64_t>(text.size()) * 8};

    for (int shift{56}; shift >= 0; shift -= 8)
    {
        padded.push_back(static_cast<char>((bit_length >> static_cast<unsigned>(shift)) & 0xFF));
    }

    return padded;
}

/**
 * @brief Returns the message schedule of one 64-byte block: its sixteen big-endian words and the 48 SHA-256 derives
 *        from them.
 * @param block The block.
 * @return The 64 words.
 */
inline std::array<std::uint32_t, 64> schedule_of(const std::string_view block)
{
    std::array<std::uint32_t, 64> schedule{};

    for (std::size_t word{0}; word < 16; ++word)
    {
        for (std::size_t byte{0}; byte < 4; ++byte)
        {
            schedule[word] = (schedule[word] << 8U) | static_cast<unsigned char>(block[(word * 4U) + byte]);
        }
    }

    for (std::size_t word{16}; word < 64; ++word)
    {
        const auto s0{
                std::rotr(schedule[word - 15], 7) ^ std::rotr(schedule[word - 15], 18) ^ (schedule[word - 15] >> 3)};

        const auto s1{
                std::rotr(schedule[word - 2], 17) ^ std::rotr(schedule[word - 2], 19) ^ (schedule[word - 2] >> 10)};

        schedule[word] = schedule[word - 16] + s0 + schedule[word - 7] + s1;
    }

    return schedule;
}

/**
 * @brief Returns a hash state after one block, its schedule run through SHA-256's 64 rounds and added to the state.
 * @param state The hash state before the block.
 * @param schedule The block's message schedule.
 * @return The hash state after it.
 */
inline std::array<std::uint32_t, 8> compressed(
        std::array<std::uint32_t, 8> state, const std::array<std::uint32_t, 64>& schedule)
{
    auto [a, b, c, d, e, f, g, h]{state};

    static constexpr std::array<std::uint32_t, 64> round_constants{
            0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
            0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
            0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
            0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
            0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
            0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
            0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
            0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2};

    for (std::size_t round{0}; round < 64; ++round)
    {
        const auto s1{std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)};

        const auto choice{(e & f) ^ (~e & g)};

        const auto t1{h + s1 + choice + round_constants[round] + schedule[round]};

        const auto s0{std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)};

        const auto majority{(a & b) ^ (a & c) ^ (b & c)};

        const auto t2{s0 + majority};

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;

    return state;
}

/**
 * @brief Returns the SHA-256 of a byte string, as hexadecimal, the digest each pinned file and table is named by.
 * @param text The bytes.
 * @return The 64 hexadecimal digits.
 */
inline std::string sha256(const std::string_view text)
{
    std::array<std::uint32_t, 8> state{0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                                       0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};

    const auto padded{padded_message(text)};

    for (std::size_t block{0}; block < padded.size(); block += 64)
    {
        const auto schedule{schedule_of(std::string_view{padded}.substr(block, 64))};

        state = compressed(state, schedule);
    }

    std::string hex{};

    for (const auto word : state)
    {
        hex += std::format("{:08x}", word);
    }

    return hex;
}

} // namespace matisse::common

#endif // MATISSE_LIBS_COMMON_INCLUDE_MATISSE_COMMON_SHA256_HPP
