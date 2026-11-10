cmake_minimum_required(VERSION 3.20.6)
project(matisse_python)


# scikit-build-core finds the interpreter and nanobind, both pinned in pyproject.toml; the stable ABI of Python 3.12
# gives one wheel per platform for every later CPython.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# NOMINSIZE keeps matisse's -O2 for the module and nanobind's conversions, which turn every id into a Python integer,
# where nanobind would optimize them for size.
nanobind_add_module(_core
        STABLE_ABI
        NB_STATIC
        NOMINSIZE
        src/module.cpp
)

target_link_libraries(_core
        PRIVATE
        matisse_encode
        matisse_lookup
        matisse_validate
)

install(TARGETS _core LIBRARY DESTINATION matisse_tokenizer)

# The widest window of each pinned tokenizer's report, as matisse's bench decides them: validate()'s default of five,
# and three for o200k, whose 30 classes take width five to minutes; registry.py names the same widths.
set(MATISSE_PYTHON_REPORTS gpt2=5 cl100k=5 o200k=3 llama3=5)

# The reports the package ships, decided once at build time by the same libraries the module links, so that a process
# reading a pinned vocabulary skips validate().
add_executable(matisse_python_reports src/write_reports.cpp)

target_link_libraries(matisse_python_reports
        PRIVATE
        matisse_lookup
        matisse_validate
)

set(MATISSE_PYTHON_REPORT_DIR ${CMAKE_CURRENT_BINARY_DIR}/reports)

set(MATISSE_PYTHON_REPORT_FILES "")

foreach (entry IN LISTS MATISSE_PYTHON_REPORTS)
    string(REPLACE "=" ";" parts ${entry})
    list(GET parts 0 name)
    list(APPEND MATISSE_PYTHON_REPORT_FILES ${MATISSE_PYTHON_REPORT_DIR}/${name}.report)
endforeach ()

add_custom_command(
        OUTPUT ${MATISSE_PYTHON_REPORT_FILES}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${MATISSE_PYTHON_REPORT_DIR}
        COMMAND matisse_python_reports ${MATISSE_PYTHON_REPORT_DIR} ${MATISSE_PYTHON_REPORTS}
        DEPENDS matisse_python_reports
        COMMENT "Deciding the reports of the pinned tokenizers"
        VERBATIM)

add_custom_target(matisse_python_report_files ALL DEPENDS ${MATISSE_PYTHON_REPORT_FILES})

install(FILES ${MATISSE_PYTHON_REPORT_FILES} DESTINATION matisse_tokenizer/reports)
