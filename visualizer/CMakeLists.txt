cmake_minimum_required(VERSION 3.16)  # Updated for FetchContent

# Include FetchContent module
include(FetchContent)

# Fetch SFML
FetchContent_Declare(
    SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 2.6.1  # Use specific stable version
)

# Configure SFML options before making it available
set(SFML_BUILD_AUDIO OFF CACHE BOOL "Build audio module" FORCE)
set(SFML_BUILD_NETWORK OFF CACHE BOOL "Build network module" FORCE)
# Keep graphics, window, and system modules


# Make SFML available
FetchContent_MakeAvailable(SFML)

add_executable(ai_visualizer
    visualizer_main.cpp
    plotter.hpp
    plotter.cpp
    vis_linear_regression.cpp
    vis_convolution.cpp
    ../src/models/linear_regression.cpp
)

# Link SFML libraries to main executable
target_link_libraries(ai_visualizer 
    sfml-graphics 
    sfml-window 
    sfml-system
)

target_include_directories(ai_visualizer PRIVATE ../include)

# For Windows: copy SFML DLLs to output directory
if(WIN32 AND BUILD_SHARED_LIBS)
    add_custom_command(
        TARGET ai_visualizer POST_BUILD
        COMMENT "Copy SFML DLLs"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:sfml-graphics> $<TARGET_FILE_DIR:ai_visualizer>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:sfml-window> $<TARGET_FILE_DIR:ai_visualizer>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:sfml-system> $<TARGET_FILE_DIR:ai_visualizer>
        VERBATIM
    )
endif()