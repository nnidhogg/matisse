cmake_minimum_required(VERSION 3.20.6)
cmake_policy(SET CMP0097 NEW)
project(matisse_common)

# The tests' and the probes' shared helpers, header-only; the target lives in the build tree alone, neither installed
# nor exported.
add_library(${PROJECT_NAME}
        INTERFACE
)

target_include_directories(${PROJECT_NAME}
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(${PROJECT_NAME}
        INTERFACE
        cxx_std_23
)
