# NeuralNetworkTextGenerator
built a feed forward neural network that generate text from a shakespear text

make sure to move code to a wsl(ubuntu environment)
g++ -std=c++17 -O3 -fopenmp -o benchmark TextProcessor.cpp transformerModel.cpp benchmark.cpp
./benchmark