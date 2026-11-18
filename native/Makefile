CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fPIC -Wall -Wextra
OUT := ../src/hybrid_alife/physics

all: $(OUT)/liblbm.so

$(OUT)/liblbm.so: lbm.cpp
	$(CXX) $(CXXFLAGS) -shared -o $@ $<

clean:
	rm -f $(OUT)/*.so
