all:
	mpicxx -std=c++0x -Wall -o logistic_regression logistic_regression.cpp
debug:
	mpicxx -std=c++0x Wall -g -o logistic_regression logistic_regression.cpp
clean:
	rm -fr ./logistic_regression
