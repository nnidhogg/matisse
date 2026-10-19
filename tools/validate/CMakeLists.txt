cmake_minimum_required(VERSION 3.20.6)
project(matisse_validate_cli)

# The command: matisse-validate FILE, the loaders and the validator's report behind one program.
add_executable(${PROJECT_NAME} src/main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME matisse-validate)

# The lookup finds the pattern and the vendor's special tokens a rank file is read under.
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        matisse_lookup
        matisse_validate
)

if (MATISSE_BUILD_TESTS)
    add_executable(${PROJECT_NAME}_tests
            tests/main_test.cpp
    )

    target_link_libraries(${PROJECT_NAME}_tests
            PRIVATE
            GTest::gtest_main
    )

    add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)

    # The command's own tests run the program built beside them, on the files beside them.
    add_dependencies(${PROJECT_NAME}_tests ${PROJECT_NAME})

    target_compile_definitions(${PROJECT_NAME}_tests
            PRIVATE
            SOURCE_DIR="${MATISSE_SOURCE_DIR}"
            MATISSE_VALIDATE="$<TARGET_FILE:${PROJECT_NAME}>"
    )
endif ()
