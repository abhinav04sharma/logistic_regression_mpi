all:
	mpicxx -std=c++0x -O2 -Wall -o logistic_regression logistic_regression.cpp
debug:
	mpicxx -std=c++0x -O2 -Wall -g -o logistic_regression logistic_regression.cpp
clean:
	rm -fr ./logistic_regression
