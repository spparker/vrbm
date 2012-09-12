CC=nvcc
CFLAGS=-c
NFLAGS=-arch=sm_21

GLINK=-lGL -lglut
CLINK=-L/usr/local/cuda/lib64 -lcurand


all: vrbm

vrbm: matrix.o rbm.o
	$(CC) $(NFLAGS) 2vrbm.cu matrix.o rbm.o -o 2vrbm $(GLINK) $(CLINK)

matrix.o: matrix.cpp
	$(CC) $(CFLAGS) matrix.cpp

rbm.o: rbm.cu
	$(CC) $(CFLAGS) rbm.cu

clean:
	rm -rf *o 2vrbm
