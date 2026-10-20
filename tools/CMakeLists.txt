# The validator's command line ships with the libraries, as munch-audit ships with munch.
add_subdirectory(validate)

# The probes back the figures docs/benchmarks.md quotes; a consumer building matisse as a dependency needs neither them
# nor their corpora, so they follow the same contract as the tests.
if (MATISSE_BUILD_TESTS)
    add_subdirectory(probes)
endif ()
