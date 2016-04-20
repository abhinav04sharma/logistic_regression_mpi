#include <stdio.h>
#include <math.h>

#include <iostream>
#include <sstream>

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <iterator>

#include <algorithm>
#include <random>
#include <chrono>

#include <cassert>


void split(std::vector<std::string> &splits,
           std::string str,
           const std::string delim)
{
    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delim)) != std::string::npos) {
        token = str.substr(0, pos);
        splits.push_back(token);
        str.erase(0, pos + delim.length());
    }
}

bool read_training_data(const char *file_name,
                        const char *delim,
                        std::vector<std::vector<double> > &training,
                        std::vector<std::vector<double> > &validation,
                        std::unordered_set<double> &label_set)
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
        label_set.insert(feat[0]);

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

double hypothesis(const std::vector<double> &weights,
                  const std::vector<double> &feats)
{
    double result = weights[0];
    for (size_t i = 1; i < feats.size(); ++i) {
        result += feats[i] * weights[i];
    }
    double sigmoid =  (double) 1 / (1 + exp(-1 * result));
    assert(sigmoid >= 0 && sigmoid <= 1);
    return sigmoid;
}

double predict(const std::unordered_map<double, std::vector<double> > &model,
               const std::unordered_set<double> &label_set,
               const std::vector<double> &feats)
{
    // binary classification
    if (model.size() == 1) {
        std::unordered_set<double>::const_iterator lit = label_set.begin();
        const double label1 = *(lit++);
        const double label2 = *lit;
        const std::unordered_map<double, std::vector<double> >::const_iterator mit = model.find(label1);
        assert(mit != model.end());

        return hypothesis(mit->second, feats) > 0.5 ? label1 : label2;
    }

    double max_hyp = -1;
    double max_label;
    for (std::unordered_map<double, std::vector<double> >::const_iterator it = model.begin();
         it != model.end();
         ++it)
    {
        double hyp = hypothesis(it->second, feats);
        if (hyp > max_hyp) {
            max_hyp = hyp;
            max_label = it->first;
        }
    }
    assert(max_hyp != -1);

    return max_label;
}

double cost(std::vector<std::vector<double> >::const_iterator feats,
            const std::vector<double> &weights,
            const size_t batch_size,
            const size_t dimension,
            const double true_label)
{
    double cost = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        double label = (*feats)[0] == true_label ? 1 : 0;
        // TODO: can this be simplified?
        if (dimension == 0) 
            cost += (hypothesis(weights, *feats) - label);
        else
            cost += (hypothesis(weights, *feats) - label) * (*feats)[dimension];
        ++feats;
    }
    return cost;
}

// if batch_size is equal to training data, it performs batch gradient decent
// if batch_size is equal to 1, it performs stochastic gradient decent
// if batch_size is in between, it performs mini-batch gradient decent
std::vector<double> binary_logistic_regression(std::vector<std::vector<double> > &feats,
                                               const double learning_rate,
                                               const double reg_param,
                                               const size_t batch_size,
                                               int data_passes,
                                               const double true_label = 1)
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
                double cst = cost(feats_batch, weights, num_examples, j, true_label);

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
        }
        --data_passes;
    }

    return weights;
}

void logistic_regression(std::vector<std::vector<double> > &feats, 
                         const std::unordered_set<double> &label_set,
                         std::unordered_map<double, std::vector<double> > &model,
                         const double learning_rate,
                         const double reg_param,
                         const size_t batch_size,
                         int data_passes)
{
    // binary classification problem
    if (label_set.size() == 2) {
        const std::unordered_set<double>::const_iterator label_itr = label_set.begin();
        const double label = *label_itr;
        model[label] = binary_logistic_regression(feats, learning_rate, reg_param, batch_size, data_passes, label);
        return;
    }

    for (std::unordered_set<double>::const_iterator label_itr = label_set.begin();
         label_itr != label_set.end();
         ++label_itr)
    {
        const double label = *label_itr;
        std::cout << std::endl << "Gradient decent for label: " << label << std::endl;
        model[label] = binary_logistic_regression(feats, learning_rate, reg_param, batch_size, data_passes, label);
    }
}

std::unordered_map<double, double> fscore(const std::vector<std::vector<double> > &validation,
                                          const std::unordered_set<double> &label_set,
                                          const std::unordered_map<double, std::vector<double> > &model)
{
    std::unordered_map<double, double> ret(model.size());
    std::unordered_map<double, std::unordered_map<double, double> > confusion_matrix;

    for (size_t i = 0; i < validation.size(); ++i) {
        double pred_label = predict(model, label_set, validation[i]);
        double true_label = validation[i][0];
        confusion_matrix[true_label][pred_label] += 1;
    }

    for (std::unordered_map<double, std::vector<double> >::const_iterator it = model.begin();
         it != model.end();
         ++it)
    {
        double label = it->first;
        double tp = confusion_matrix[label][label];
        double total_gold = 0;
        double total_pred = 0;
        for (std::unordered_set<double>::const_iterator lit = label_set.begin(); lit != label_set.end(); ++lit) {
            total_gold += confusion_matrix[label][*lit];
            total_pred += confusion_matrix[*lit][label];
        }

        double precision = tp / total_pred;
        double recall = tp / total_gold;

        ret[label] = (2 * precision * recall) / (precision + recall);
    }

    return ret;
}

void usage() {
    std::cout << "logistic_regression <training file> <delimiter> <learning rate> <regularization parameter> " <<
        "[<data passes (-1 for convergence)> <batch size>]" << std::endl;
}

int main(int argc,
         char * argv[])
{
    if (argc < 5 || std::string(argv[1]) == std::string("-h")) {
        usage();
        return 1;
    }

    std::vector<std::vector<double> > training; // TODO: can be optimized, we can count the number of feats and allocate accordingly
    std::vector<std::vector<double> > validation; // TODO: can be optimized, we can count the number of feats and allocate accordingly

    std::unordered_set<double> label_set;
    std::unordered_map<double, std::vector<double> > model;

    char *training_file = argv[1];
    char *delimiter = argv[2];

    // read training data
    std::cout << "Reading training data ... ";
    read_training_data(training_file, delimiter, training, validation, label_set);
    std::cout << "[Done]" << std::endl;

    double learning_rate = std::stod(std::string(argv[3]));
    double reg_param = std::stod(std::string(argv[4]));
    int data_passes = -1;
    size_t batch_size = training.size();

    if (argc >= 6)
        data_passes = std::stoi(std::string(argv[5]));
    if (argc >= 7)
        batch_size = std::stoul(std::string(argv[6]));

    std::cout << std::endl
        << "***Info***" << std::endl
        << "Training Data File: " << training_file << std::endl
        << "Learning Rate: " << learning_rate << std::endl
        << "Regularization Parameter: " << reg_param << std::endl
        << "Data Passes: " << data_passes << std::endl
        << "Batch Size: " << batch_size << std::endl
        << "Num Labels: " << label_set.size() << std::endl
        << "**********" << std::endl;


    // logistic regression
    logistic_regression(training, label_set, model, learning_rate, reg_param, batch_size, data_passes);

    // print the f-score(s)
    const std::unordered_map<double, double> fsc = fscore(validation, label_set, model);
    std::cout << std::endl << std::endl;
    std::cout << "F-Score(s):" << std::endl;
    std::cout << "Label\tScore" << std::endl;
    for (std::unordered_map<double, double>::const_iterator it = fsc.begin(); it != fsc.end(); ++it) {
        std::cout << it->first << "\t" << it->second << std::endl;
    }

    return 0;
}
