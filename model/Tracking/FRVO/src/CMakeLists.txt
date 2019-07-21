cmake_minimum_required(VERSION 3.12)
project(src)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(src
        Agent.cpp
        Agent.h
        Definitions.h
        KdTree.cpp
        KdTree.h
        Motion.h
        Obstacle.cpp
        Obstacle.h
        RVO.h
        RVOSimulator.cpp
        RVOSimulator.h
        Vector2.h demo.cpp)
