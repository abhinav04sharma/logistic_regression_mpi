all:
	g++ -std=c++11 -fopenmp -o logistic_regression logistic_regression.cpp
debug:
	g++ -std=c++11 -g -fopenmp -o logistic_regression logistic_regression.cpp
clean:
	rm -fr ./logistic_regression
