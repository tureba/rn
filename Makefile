
CFLAGS=-lm -pipe -Wall -Wextra -std=c99 -ggdb -pg

all: neural

neural: rn.o

clean:
	-rm -f neural *.o
