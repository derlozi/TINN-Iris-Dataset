cmake_minimum_required(VERSION 3.10)
project(FirstTINN C)

set(CMAKE_C_STANDARD 99)
include_directories(C:/Users/loren/Documents/Programmieren/C)
include_directories(C:/Users/loren/Documents/Programmieren/C/tinn-master)
add_subdirectory(tinn-master)

add_executable(FirstTINN main.c)
target_link_libraries(FirstTINN Tinn)