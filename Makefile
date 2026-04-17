# Makefile for C Auto-tuning Evaluator

CC = gcc
CFLAGS = -shared -fPIC -O3 -march=native -ffast-math -Wall -fopenmp
LDFLAGS = -lm -lgsl -lgslcblas -llbfgs

SRCS = src/csv.c src/random.c src/curvefit.c src/evaluator.c src/minicsv.c
TARGET = libevaluator.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
