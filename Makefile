
CFLAGS=-lm -pipe -Wall -Wextra -std=c99 -ggdb -pg -fgraphite -fgraphite-identity -floop-parallelize-all

all: neural

neural: rn.o

clean:
	-rm -f neural *.o neural.c.* rn.c.*
