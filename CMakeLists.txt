cmake_minimum_required(VERSION 3.16)
project(AIFromScratchCPP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main executable
add_executable(ai_main
    include/common.hpp
    include/datasynth.hpp
    include/models/linear_regression.hpp
    include/models/logistic_regression.hpp
    include/models/knearest_neighbors.hpp
    src/main.cpp
    src/models/linear_regression.cpp
    src/models/logistic_regression.cpp
    src/models/knearest_neighbors.cpp
)

# Include headers
target_include_directories(ai_main PRIVATE include)

# Add tests
enable_testing()
add_subdirectory(tests)

# Add visualization option
option(ENABLE_VISUALIZATION "Enable SFML visualization" ON)
if(ENABLE_VISUALIZATION)
    add_subdirectory(visualizer)
endif()
