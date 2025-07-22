cmake_minimum_required(VERSION 3.10)
project(AIFromScratchCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main executable
add_executable(ai_main
    include/models/linear_regression.hpp
    src/main.cpp
    src/models/linear_regression.cpp
)

# Include headers
target_include_directories(ai_main PRIVATE include)

# Add tests
enable_testing()
add_subdirectory(tests)
