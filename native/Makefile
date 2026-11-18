CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fPIC -Wall -Wextra
OUT := ../src/hybrid_alife/physics
LIBS := $(patsubst %.cpp,$(OUT)/lib%.so,$(wildcard *.cpp))

all: $(LIBS)

$(OUT)/lib%.so: %.cpp
	$(CXX) $(CXXFLAGS) -shared -o $@ $<

clean:
	rm -f $(OUT)/*.so
