all:
	mpicxx -std=c++0x -o logistic_regression logistic_regression.cpp
debug:
	mpicxx -std=c++0x -g -o logistic_regression logistic_regression.cpp
clean:
	rm -fr ./logistic_regression
