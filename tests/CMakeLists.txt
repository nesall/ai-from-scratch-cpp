add_executable(ai_tests
    test_main.cpp
    test_linear_regression.cpp
    test_logistic_regression.cpp
    test_knearest_neighbors.cpp
    test_naivebayes_classifier.cpp
    test_decision_tree.cpp
    ../src/models/linear_regression.cpp
    ../src/models/logistic_regression.cpp
    ../src/models/knearest_neighbors.cpp
    ../src/models/naivebayes_classifier.cpp
    ../src/models/decision_tree.cpp

)

target_include_directories(ai_tests PRIVATE ../include)
add_test(NAME AITests COMMAND ai_tests)
