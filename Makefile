CC = gcc
CFLAGS = -shared -fPIC -O2 -Wall -fopenmp -Isrc
LDFLAGS = -lm -lgsl -lgslcblas -llbfgs

SRCS = src/csv.c src/curvefit.c src/evaluator.c src/historical_cache.c \
       src/minicsv.c

TARGET = libevaluator.so

.PHONY: all clean check

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

check: $(TARGET)
	@echo "checking for undefined symbols..."
	@! nm -u $(TARGET) | grep -E ' (curve_|history_|get_regression_|minicsv_|csv_)' \
	  || { echo "ERROR: unresolved project symbols above"; exit 1; }
	@echo "OK"

clean:
	rm -f $(TARGET)
