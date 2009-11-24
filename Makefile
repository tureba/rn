
CFLAGS=-lm -pipe -Wall -Wextra -std=c99 -ggdb

all: NEURAL

NEURAL: rn.o

clean:
	-rm -f NEURAL *.o
