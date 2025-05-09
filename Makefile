CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp
LDFLAGS = -fopenmp

# Targets
all: transformer shakespeare

# Transformer model with benchmarks
transformer: transformerModel.cpp
	$(CXX) $(CXXFLAGS) -o transformer transformer-model.cpp $(LDFLAGS)

# Shakespeare next word prediction
shakespeare: shakespearePrediction.cpp
	$(CXX) $(CXXFLAGS) -o shakespeare shakespeare-prediction.cpp $(LDFLAGS)

clean:
	rm -f transformer shakespeare *.o

.PHONY: all clean