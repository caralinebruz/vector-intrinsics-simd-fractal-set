
# MAC M1 -- installed opencv package via brew install opencv
# LDFLAGS:=$(shell pkg-config --cflags --libs /opt/homebrew/Cellar/opencv/*/lib/pkgconfig/opencv4.pc)

# INTEL MAC
# LDFLAGS:=$(shell pkg-config --cflags --libs /usr/local/Cellar/opencv/*/lib/pkgconfig/opencv4.pc)

# CIMS Server, need to load module
# module loac gcc-12.2
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv)
OPENCV_LIBS := $(shell pkg-config --libs opencv)
LDFLAGS := $(OPENCV_LIBS) $(OPENCV_CFLAGS) -lstdc++ -lm 

CC:=gcc -mavx -march=native -O3
CFLAGS:=-std=gnu++11
SRC_FILES:=$(wildcard ./*.cpp)
OBJ_FILES:=$(patsubst %.cpp,obj/%.o,$(SRC_FILES))

all: burningship

burningship: $(OBJ_FILES)
	$(CC) $(LDFLAGS) -o $@ $^

obj/%.o: %.cpp
	$(CC) $(LDFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	rm burningship
	rm obj/*.o
