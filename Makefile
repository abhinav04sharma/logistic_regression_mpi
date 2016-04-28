all:
	g++ -std=c++0x -Wall -fopenmp -o logistic_regression logistic_regression.cpp
debug:
	g++ -std=c++0x -Wall -g -fopenmp -o logistic_regression logistic_regression.cpp
clean:
	rm -fr ./logistic_regression
