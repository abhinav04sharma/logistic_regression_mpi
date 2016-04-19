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

bool read_training_data(const char *file_name, const char *delim, std::vector<std::vector<double> > &training,
        std::vector<std::vector<double> > &validation)
{
    FILE *fp = fopen(file_name, "r");
    char *line = NULL;
    size_t len  = 0;
    ssize_t read = 0;

    if (fp == NULL)
        return false;

    size_t count = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        std::vector<std::string> splits;
        split(splits, std::string(line), std::string(delim));

        // extract features
        std::vector<double> feat;
        for (size_t i = 0; i < splits.size(); ++i) {
            feat.push_back(std::stod(splits[i], NULL));
        }

        // case: 80% training, 20% validation
        if (count >= 8) validation.push_back(feat);
        else training.push_back(feat);

        // clean up
        free(line);
        line = NULL;

        count = ((count + 1) % 10);
    }

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
    double sigmoid =  (double) 1 / (1 + exp(-1 * result));
    assert(sigmoid >= 0 && sigmoid <= 1);
    return sigmoid;
}

double cost(std::vector<std::vector<double> >::const_iterator feats, const std::vector<double> &weights, 
        const size_t batch_size, const size_t dimension)
{
    double cost = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        // TODO: can this be simplified?
        if (dimension == 0) 
            cost += (hypothesis(weights, *feats) - (*feats)[0]);
        else
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
    size_t count = 0;
    bool is_batch_gradient_decent = batch_size == feats.size();

    // gradient decent
    while(data_passes != 0) {
        std::cout << "Iteration: " << count++ << std::endl;

        std::vector<std::vector<double> >::const_iterator feats_batch = feats.begin();
        std::vector<double> temp_weights(weights.size());

        // for each batch of data
        while (feats_batch < feats.end()) {
            for (size_t j = 0; j < temp_weights.size(); ++j) {
                // calculate cost for the batch
                const size_t num_examples = std::min(batch_size, (size_t) (feats.end() - feats_batch));
                double cst = cost(feats_batch, weights, num_examples, j);

                // regularization is not applied to the zeroth weight
                // TODO: can this be simplified?
                if (j > 0) {
                    temp_weights[j] = weights[j] * (1 - learning_rate * reg_param / num_examples) - learning_rate / num_examples * cst;
                } else {
                    temp_weights[j] = weights[j] - learning_rate / num_examples * cst;
                }
            }

            // advance by batch size;
            feats_batch += batch_size;

            // convergence
            if (temp_weights == weights) {
                return weights;
            }

            std::copy(temp_weights.begin(), temp_weights.end(), weights.begin());
            assert(temp_weights == weights);
        }

        // shuffle the data
        if (!is_batch_gradient_decent) {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            shuffle(feats.begin(), feats.end(), std::default_random_engine(seed));
            //random_shuffle(feats.begin(), feats.end());
        }
        --data_passes;
    }

    return weights;
}

double fscore(std::vector<std::vector<double> > &validation, std::vector<double> weights)
{
    size_t tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < validation.size(); ++i) {
        double hyp = hypothesis(weights, validation[i]);
        if (hyp > 0.5) {
            if (validation[i][0] == 1) ++tp;
            else ++fp;
        } else {
            if (validation[i][0] == 1) ++fn;
        }
    }

    double precision = (double) tp / (tp + fp);
    double recall = (double) tp / (tp + fn);

    return (2 * precision * recall) / (precision + recall);
}

void usage() {
    std::cout << "logistic_regression <training file> <delimiter> <learning rate> <regularization parameter> " <<
        "[<data passes (-1 for convergence)> <batch size>]" << std::endl;
}

int main(int argc, char * argv[])
{
    if (argc < 5 || std::string(argv[1]) == std::string("-h")) {
        usage();
        return 1;
    }

    std::vector<std::vector<double> > training; // TODO: can be optimized, we can count the number of feats and allocate accordingly
    std::vector<std::vector<double> > validation; // TODO: can be optimized, we can count the number of feats and allocate accordingly

    char *training_file = argv[1];
    char *delimiter = argv[2];

    // read training data
    read_training_data(training_file, delimiter, training, validation);
    std::cout << "Done reading training data ... " << std::endl;

    double learning_rate = std::stod(std::string(argv[3]));
    double reg_param = std::stod(std::string(argv[4]));
    int data_passes = -1;
    size_t batch_size = training.size();

    if (argc == 6)
        data_passes = std::stoi(std::string(argv[5]));
    if (argc == 7)
        batch_size = std::stoul(std::string(argv[6]));

    std::cout 
        << "***Info***" << std::endl
        << "Training Data File: " << training_file << std::endl
        << "Learning Rate: " << learning_rate << std::endl
        << "Regularization Parameter: " << reg_param << std::endl
        << "Data Passes: " << data_passes << std::endl
        << "Batch Size: " << batch_size << std::endl << std::endl;

    // logistic regression
    std::vector<double> weights = logistic_regression(training, learning_rate, reg_param, batch_size, data_passes);

    std::cout << "F-Score: " << fscore(validation, weights) << std::endl;
    return 0;
}
