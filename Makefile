CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan

build: src/main.cpp
	g++ $(CFLAGS) -o main src/main.cpp $(FILES) $(LDFLAGS)

.PHONY: launch clean

launch: build
	./main

clean:
	rm -f main