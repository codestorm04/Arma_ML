cmake_minimum_required(VERSION 3.5)
project(Machine_Learning_CPP)

set(CMAKE_CXX_STANDARD 14)

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(src SRC_DIR)

# 指定生成目标
add_executable(Machine_Learning_CPP ${SRC_DIR})
