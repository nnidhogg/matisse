#include <cstdlib>
#include <iostream>
#include <string_view>

namespace
{
/**
 * @brief The usage text: the command line the validator takes and what each option does.
 */
constexpr std::string_view usage_text{R"(usage: matisse-validate [options] FILE

Reads a tokenizer definition, a Hugging Face tokenizer.json or a tiktoken rank file, and reports which windows of its
pre-tokenizer's classes certify a cut, at which origin and with which right context, under the factorization and the
policy certificate.

  --tiktoken PATTERN    read FILE as a tiktoken rank file whose pre-tokenizer is PATTERN, gpt2 or cl100k, since the
                        file names none
  --windows N           the widest window decided, 4 unless given
  --input FILE          measure each inventory's certified-anchor supply on this file: the anchors, per kibibyte, the
                        gaps between them, the share of the pre-token boundaries they reach and the longest piece
  --json                one JSON document instead of text
)"};

} // namespace

int main()
{
    std::cout << usage_text;

    return EXIT_SUCCESS;
}
