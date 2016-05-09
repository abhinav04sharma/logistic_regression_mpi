# Logistic Regression on MPI
Utility to run logistic regression over MPI. It was built to run gradient descent quickly over data with high dimensionality. The idea is to split dimensions across MPI workers and process gradients in parallel. The implementation consists of two types of MPI processes:
* Parameter Server: Responsible for distribution and collection of weights
and data batches
* Workers: Responsible for calculating gradient for a range of dimensions

There is a single parameter server and there can be one or more worker (i.e. it needs at least two MPI workers to run).

It uses 20% of the traning data for validation. It supports two modes:
* Synchronous: In this mode the parameter server does not begin calculation with the next batch until all the workers return their gradients for the current batch. This is mathematically sound.
* Asynchronous: In this mode the parameter server sends latest weights to a ready worker regardless of whether the previous batch is completed by all workers.

Asynchronous mode is expected to have a smaller runtime especially when run on a heterogenous cluster, it may lead to a lower accuray though.

To build an executable for simply run `make`.

The executable takes the following command line parameters:

`logistic_regression <training file> <delimiter> <learning rate> <regularization parameter> <sync/async> <data passes> [<batch size>]`

Batch size is optional and when not provided the entire data set is used (which is basically batch gradient descent).

A sample PBS script is provided to run the code on a cluster.
