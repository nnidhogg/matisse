# The loaders fed arbitrary bytes and the encoder fed arbitrary well-formed UTF-8. The contract is the libraries' own: a
# std::invalid_argument is a loader's answer to a file it cannot take exactly, and a crash, a sanitizer report or a hang
# is a finding, as is a violated property. Each harness is its own target so that each corpus holds inputs that decode
# as its format and not as another's, and each is seeded from its own directory under seeds: a small rank file, a GPT-2
# pair in one file cut at a NUL byte, and texts each led by the two bytes that choose the encoder and the chunk count.
#
# The invariant check and the files the loaders read, which the harnesses share, their header under
# include/matisse/tools/fuzz.
add_library(matisse_fuzz
        src/harness.cpp
)

target_include_directories(matisse_fuzz
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(matisse_tiktoken_fuzzer
        src/tiktoken_fuzzer.cpp
)

target_link_libraries(matisse_tiktoken_fuzzer
        PRIVATE
        matisse_fuzz
        matisse_load
)

# fuzzer rather than the tree-wide fuzzer-no-link: this target carries libFuzzer's driver and its main().
target_link_options(matisse_tiktoken_fuzzer
        PRIVATE
        -fsanitize=fuzzer
)

add_executable(matisse_gpt2_fuzzer
        src/gpt2_fuzzer.cpp
)

target_link_libraries(matisse_gpt2_fuzzer
        PRIVATE
        matisse_fuzz
        matisse_load
)

target_link_options(matisse_gpt2_fuzzer
        PRIVATE
        -fsanitize=fuzzer
)

add_executable(matisse_encoder_fuzzer
        src/encoder_fuzzer.cpp
)

target_link_libraries(matisse_encoder_fuzzer
        PRIVATE
        matisse_cl100k
        matisse_encode
        matisse_fuzz
        matisse_gpt2
        matisse_llama3
        matisse_o200k
)

target_link_options(matisse_encoder_fuzzer
        PRIVATE
        -fsanitize=fuzzer
)
