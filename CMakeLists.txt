cmake_minimum_required(VERSION 3.0)
project(conv)
find_package(CUDA REQUIRED)
enable_language(CUDA)

add_executable(main main.cu)
