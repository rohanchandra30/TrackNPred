OBJS = Agent.o KdTree.o Obstacle.o RVOSimulator.o
CC = g++
DEBUG = -g
OMP = -fopenmp
CFLAGS = -Wall -c $(DEBUG) $(OMP)

libRVO.a : $(OBJS)
	ar ru libRVO.a $(OBJS)
	ranlib libRVO.a
	rsync RVO.h RVOSimulator.h Vector2.h ../include/
	rsync libRVO.a ../lib/

Agent.o : Agent.h Definitions.h KdTree.h Obstacle.h RVOSimulator.h Vector2.h Agent.cpp
	$(CC) $(CFLAGS) Agent.cpp

KdTree.o : Agent.h Definitions.h KdTree.h Obstacle.h RVOSimulator.h Vector2.h KdTree.cpp
	$(CC) $(CFLAGS) KdTree.cpp

Obstacle.o : Definitions.h Obstacle.h RVOSimulator.h Vector2.h Obstacle.cpp
	$(CC) $(CFLAGS) Obstacle.cpp

RVOSimulator.o : Agent.h KdTree.h Obstacle.h RVOSimulator.h Vector2.h RVOSimulator.cpp
	$(CC) $(CFLAGS) RVOSimulator.cpp

clean:
	rm -f *.o libRVO.a ../include/RVO.h ../include/RVOSimulator.h ../include/Vector2.h ../lib/libRVO.a
