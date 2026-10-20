cmake_minimum_required(VERSION 3.20.6)
project(matisse_probes)

# The commit the bench reports, regenerated every build so it cannot go stale: once at configure time so the header
# exists for the first compile, then again on every build to refresh it, as munch's benchmarks record theirs.
set(MATISSE_PROVENANCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

set(MATISSE_PROVENANCE_HEADER ${MATISSE_PROVENANCE_DIR}/matisse/tools/probes/provenance.hpp)

execute_process(COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${MATISSE_SOURCE_DIR}
        -DTEMPLATE=${CMAKE_CURRENT_SOURCE_DIR}/provenance.hpp.in
        -DOUTPUT=${MATISSE_PROVENANCE_HEADER}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/provenance.cmake)

add_custom_target(matisse_provenance
        BYPRODUCTS ${MATISSE_PROVENANCE_HEADER}
        COMMAND ${CMAKE_COMMAND}
                -DSOURCE_DIR=${MATISSE_SOURCE_DIR}
                -DTEMPLATE=${CMAKE_CURRENT_SOURCE_DIR}/provenance.hpp.in
                -DOUTPUT=${MATISSE_PROVENANCE_HEADER}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/provenance.cmake
        COMMENT "Recording the commit the bench is built from"
        VERBATIM)

# The corpus probe: the ids of encode() and of encode_parallel() on GPT-2's and cl100k's pinned vocabularies, held to
# the known-answer fixture's corpus digests, to encode() itself at every certified cut, and to the merge with no cache.
# Runs as a test on generated texts when the vocabulary directory is given, and on the three corpora of the test
# strategy as well when the corpus directory is given; any other corpus is named on its command line.
add_executable(matisse_corpus src/corpus.cpp)

target_link_libraries(matisse_corpus
        PRIVATE
        matisse_cl100k
        matisse_common
        matisse_encode
        matisse_gpt2
        matisse_validate
)

target_compile_definitions(matisse_corpus
        PRIVATE
        SOURCE_DIR="${MATISSE_SOURCE_DIR}"
)

# The bench times what the corpus probe checks, every figure taken after its result is held to encode()'s ids.
add_executable(matisse_bench src/bench.cpp)

add_dependencies(matisse_bench matisse_provenance)

target_include_directories(matisse_bench
        PRIVATE
        ${MATISSE_PROVENANCE_DIR}
)

target_link_libraries(matisse_bench
        PRIVATE
        matisse_cl100k
        matisse_common
        matisse_encode
        matisse_gpt2
        matisse_validate
)

# The bench's test is one pass of every single-thread row on the validator's sample, for its references and its exit.
if (MATISSE_VOCAB_DIR)
    add_test(NAME matisse_corpus COMMAND matisse_corpus ${MATISSE_VOCAB_DIR})

    add_test(NAME matisse_bench
            COMMAND matisse_bench single ${MATISSE_VOCAB_DIR} gpt2,cl100k 1
                    ${MATISSE_SOURCE_DIR}/tools/validate/tests/data/sample.txt)
endif ()

if (MATISSE_VOCAB_DIR AND MATISSE_CORPUS_DIR)
    add_test(NAME matisse_corpus_fixtures
            COMMAND matisse_corpus ${MATISSE_VOCAB_DIR}
                    ${MATISSE_CORPUS_DIR}/twitter.json
                    ${MATISSE_CORPUS_DIR}/munch-python@15b2863c.txt
                    ${MATISSE_CORPUS_DIR}/munch-cpp@15b2863c.txt)
endif ()
