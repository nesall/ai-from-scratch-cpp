add_executable(ai_tests
    test_main.cpp
    test_linear_regression.cpp
    ../src/models/linear_regression.cpp

)

target_include_directories(ai_tests PRIVATE ../include)
add_test(NAME AITests COMMAND ai_tests)
