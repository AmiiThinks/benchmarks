cmake_minimum_required(VERSION 3.10)
project(chinese-checkers)

# find_package(PythonLibs REQUIRED)
# include_directories(${PYTHON_INCLUDE_DIRS})
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -w -fPIC -pthread")
# set(PYTHON_EXECUTABLE "/home/zaheen/anaconda3/envs/tf2/bin/python3")

include_directories("cc"
    "utils" 
    "players"
    "solve"
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIR})

add_subdirectory(utils)
add_subdirectory(cc)
add_subdirectory(solve)
add_subdirectory(players)
add_subdirectory(game)
add_subdirectory(python2)

pybind11_add_module(ccwrapper python2/ccwrapper.cpp)
target_link_libraries(ccwrapper PRIVATE utils cc players solve)

#add_executable(solve_cc solve/main.cpp)
pybind11_add_module(solve_cc solve/main.cpp)
target_include_directories(solve_cc PUBLIC utils cc solve)
target_link_libraries(solve_cc PUBLIC utils cc solve)

#add_executable(game_cc game/Driver.cpp)
pybind11_add_module(game_cc game/Driver.cpp)
target_include_directories(game_cc PUBLIC utils cc players game)
target_link_libraries(game_cc PUBLIC utils cc players game)

