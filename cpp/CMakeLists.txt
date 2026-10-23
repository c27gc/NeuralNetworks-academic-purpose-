cmake_minimum_required(VERSION 3.16)
project(lecture_nn_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(lecture_nn STATIC
    src/lecture_nn.cpp
    "../C version/Separate Classes/Neuron Class/neuron.cpp"
)

target_include_directories(lecture_nn
    PUBLIC include
    PRIVATE "../C version/Separate Classes/Neuron Class"
)

target_compile_options(lecture_nn PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(lecture_nn_tests tests/test_lecture_nn.cpp)
target_link_libraries(lecture_nn_tests PRIVATE lecture_nn)
add_test(NAME lecture_nn_tests COMMAND lecture_nn_tests)
