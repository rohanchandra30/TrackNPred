CC = g++
DEBUG = -g
OMP = -fopenmp
CFLAGS = -Wall -c $(DEBUG) $(OMP) -I../include/
LFLAGS = -Wall $(DEBUG) -L../lib -lRVO -lgomp

all : frvo

frvo : frvo.o
	$(CC) $(LFLAGS) frvo.o -o frvo
	rsync frvo ../bin/
	
frvo.o : frvo.cpp
	$(CC) $(CFLAGS) frvo.cpp
clean:
	rm -f *.o frvo ../bin/frvo

