cmake_minimum_required(VERSION 3.20.6)
project(matisse_validate_cli)

# The command: matisse-validate FILE, the loaders and the validator's report behind one program.
add_executable(${PROJECT_NAME} src/main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME matisse-validate)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        matisse_validate
)
