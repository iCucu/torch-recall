cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_recall_online CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

file(GLOB_RECURSE SOURCES src/*.cpp)

add_executable(torch_recall_cli ${SOURCES})
target_include_directories(torch_recall_cli PRIVATE include)
target_link_libraries(torch_recall_cli "${TORCH_LIBRARIES}")
