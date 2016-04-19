#include <stdio.h>
#include <math.h>

#include <iostream>
#include <sstream>
#include <iterator>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cassert>


void split(std::vector<std::string> &splits, std::string str, const std::string delim)
{
    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delim)) != std::string::npos) {
        token = str.substr(0, pos);
        splits.push_back(token);
        str.erase(0, pos + delim.length());
    }
}

bool read_training_data(const char *file_name, const char *delim, std::vector<std::vector<double> > &feats)
{
    FILE *fp = fopen(file_name, "r");
    char *line = NULL;
    size_t len  = 0;
    ssize_t read = 0;

    if (fp == NULL)
        return false;

    std::cout << "Reading training data from: " << file_name << std::endl;

    while ((read = getline(&line, &len, fp)) != -1) {
        std::vector<std::string> splits;
        split(splits, std::string(line), std::string(delim));

        // extract features
        std::vector<double> feat;
        for (size_t i = 0; i < splits.size(); ++i) {
            feat.push_back(std::stod(splits[i], NULL));
        }

        feats.push_back(feat);

        // clean up
        free(line);
        line = NULL;
    }

    std::cout << "Done!" << std::endl;

    free(line);
    fclose(fp);
    return true;
}

double hypothesis(const std::vector<double> &weights, const std::vector<double> &feats)
{
    double result = weights[0];
    for (size_t i = 1; i < feats.size(); ++i) {
        result += feats[i] * weights[i];
    }
    return 1 / (1 + exp(-1 * result));
}

double cost(std::vector<std::vector<double> >::const_iterator feats, const std::vector<double> &weights, 
        const size_t batch_size, const size_t dimension)
{
    double cost = 0;
    // feature of 0th weight is always 1
    cost += (hypothesis(weights, *feats) - (*feats)[0]);
    for (size_t i = 1; i < batch_size; ++i) {
        cost += (hypothesis(weights, *feats) - (*feats)[0]) * (*feats)[dimension];
        ++feats;
    }
    return cost;
}

// if batch_size is equal to training data, it performs batch gradient decent
// if batch_size is equal to 1, it performs stochastic gradient decent
// if batch_size is in between, it performs mini-batch gradient decent
std::vector<double> logistic_regression(std::vector<std::vector<double> > &feats, const double learning_rate,
        const double reg_param, const size_t batch_size, int data_passes)
{
    std::vector<double> weights(feats[0].size(), 0);
    bool converge = false;
    size_t count = 0;

    assert(batch_size != feats.size() || data_passes == 0);

    // gradient decent
    while(true) {

        std::cout << "Iteration: " << count++ << std::endl;

        std::vector<std::vector<double> >::const_iterator feats_batch = feats.begin();

        // for each batch of data
        while (feats_batch < feats.end()) {
            std::vector<double> temp_weights(weights);

            for (int j = 0; j < temp_weights.size(); ++j) {
                // calculate cost for the batch
                const size_t num_examples = std::min(batch_size, (size_t) (feats.end() - feats_batch));
                double cst = cost(feats_batch, weights, num_examples, j);

                // regularization is not applied to the zeroth weight
                if (j > 0) {
                    temp_weights[j] = temp_weights[j] * (1 - learning_rate * reg_param / num_examples) - learning_rate / num_examples * cst;
                } else {
                    temp_weights[j] = temp_weights[j] - learning_rate / num_examples * cst;
                }
            }

            // advance by batch size;
            feats_batch += batch_size;

            // convergence
            if (temp_weights == weights) {
                converge = true;
                break;
            }

            std::copy(temp_weights.begin(), temp_weights.end(), weights.begin());
        }

        // either convergence or data passes have exhausted for mini-batch/stocastic decent
        if (converge || (data_passes == 0 && batch_size != feats.size())) {
            break;
        } else {
            // shuffle the data
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            shuffle(feats.begin(), feats.end(), std::default_random_engine(seed));
            --data_passes;
        }
    }

    return weights;
}

void usage() {
    std::cout << "logistic_regression <training file> <delimiter> <learning rate> <regularization parameter> [<batch size> <data passes>]"
        << std::endl;
}

int main(int argc, char * argv[])
{
    if (argc < 5 || std::string(argv[1]) == std::string("-h")) {
        usage();
        return 1;
    }

    std::vector<std::vector<double> > feats; // TODO: can be optimized, we can count the number of feats and allocate accordingly

    char *filename = argv[1];
    char *delimiter = argv[2];

    // read training data
    read_training_data(filename, delimiter, feats);

    double learning_rate = std::stod(std::string(argv[3]));
    double reg_param = std::stod(std::string(argv[4]));
    size_t batch_size = feats.size();
    int data_passes = 0;

    if (argc > 5) {
        batch_size = std::stoul(std::string(argv[5]));
        data_passes = std::stoi(std::string(argv[6]));
    }

    // logistic regression
    std::vector<double> weights = logistic_regression(feats, learning_rate, reg_param, batch_size, data_passes);

    // TODO: validation set statistics (fscore, precision etc.)
    // print the model
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << weights[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
