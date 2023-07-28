g++ -fPIC -no-pie -O3 -c -g NeuralNets.c
g++ -fPIC -no-pie -O3 -g *.o -lm -o NeuralNets

