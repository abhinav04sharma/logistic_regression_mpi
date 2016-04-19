all:
	g++ -std=c++11 -o logistic_regression logistic_regression.cpp
debug:
	g++ -std=c++11 -g -o logistic_regression logistic_regression.cpp
clean:
	rm -fr ./logistic_regression
