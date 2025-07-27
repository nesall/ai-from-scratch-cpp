add_executable(ai_tests
    test_main.cpp
    test_linear_regression.cpp
    test_logistic_regression.cpp
    test_knearest_neighbors.cpp
    test_naivebayes_classifier.cpp
    test_decision_tree.cpp
    test_random_forest.cpp
    test_svm.cpp
    test_matrix.cpp
    test_perceptron.cpp
    test_mlp.cpp
    test_mnist_mlp.cpp
    ../src/models/linear_regression.cpp
    ../src/models/logistic_regression.cpp
    ../src/models/knearest_neighbors.cpp
    ../src/models/naivebayes_classifier.cpp
    ../src/models/decision_tree.cpp
    ../src/models/random_forest.cpp
    ../src/models/svm.cpp
    ../src/matrix.cpp
    ../src/models/perceptron.cpp
    ../src/models/mlp.cpp

)

target_include_directories(ai_tests PRIVATE ../include)
add_test(NAME AITests COMMAND ai_tests)
