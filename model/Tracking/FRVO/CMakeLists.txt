cmake_minimum_required(VERSION 3.12)
project(FRVO)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(FRVO
        demo/build/demo.cpp
        demo/frvo.cpp
        src/Agent.cpp
        src/Agent.h
        src/Definitions.h
        src/demo.cpp
        src/KdTree.cpp
        src/KdTree.h
        src/Motion.h
        src/Obstacle.cpp
        src/Obstacle.h
        src/RVO.h
        src/RVOSimulator.cpp
        src/RVOSimulator.h
        src/Vector2.h)
