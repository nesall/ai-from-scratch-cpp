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
    include/models/naivebayes_classifier.hpp
    include/models/decision_tree.hpp
    include/models/random_forest.hpp
    include/models/svm.hpp
    include/matrix.hpp
    include/models/perceptron.hpp
    include/models/mlp.hpp
    include/models/cnn.hpp
    include/models/rnn.hpp
    include/models/optimizers.hpp
    include/models/regularization.hpp
    include/models/q_learning.hpp
    include/models/kmeans.hpp
    include/models/pca.hpp
    src/main.cpp
    src/models/linear_regression.cpp
    src/models/logistic_regression.cpp
    src/models/knearest_neighbors.cpp
    src/models/naivebayes_classifier.cpp
    src/models/decision_tree.cpp 
    src/models/random_forest.cpp 
    src/models/svm.cpp 
    src/matrix.cpp 
    src/models/perceptron.cpp 
    src/models/mlp.cpp 
    src/models/cnn.cpp 
    src/models/rnn.cpp 
    src/models/q_learning.cpp
    src/models/kmeans.cpp
    src/models/pca.cpp
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
