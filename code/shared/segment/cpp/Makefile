
CC=gcc
CXX=g++
CFLAGS=-c -Wall -O3
#LDFLAGS=-lm

FILES=fukuda_tools data_tools io_tools segmentation_tools
OBJECTS=$(FILES:=.o)

EXECUTABLES=generate_data fukuda_breakpoints fukuda_segmentation
LIBRARIES=

all: $(EXECUTABLES) $(LIBRARIES)

generate_data: generate_data.o $(OBJECTS)
	$(CXX) $^ -o $@

fukuda_breakpoints: fukuda_breakpoints.o $(OBJECTS)
	$(CXX) $^ -o $@

fukuda_segmentation: fukuda_segmentation.o $(OBJECTS)
	$(CXX) $^ -o $@ 

# clear out all suffixes
.SUFFIXES:
# clear out all suffixes
.SUFFIXES: .o .cpp

# set up a generic c++ alias
.cpp.o:
	$(CC) $(CFLAGS) $<

# get this shit out of here!
clean:
	rm -rf $(OBJECTS) $(EXECUTABLES) $(LIBRARIES)
