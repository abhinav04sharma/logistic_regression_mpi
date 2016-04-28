#include <stdio.h>
#include <math.h>

#include <iostream>
#include <sstream>

#include <string>
#include <string.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <iterator>

#include <omp.h>
#include <mpi.h>

#include <algorithm>
#include <random>

#include <cassert>


int rank, namelen, numprocs;
char processor_name[MPI_MAX_PROCESSOR_NAME];
double learning_rate;
double reg_param;


enum TAGS {
    DATA_PASSES,

    TRUE_LABEL,
    NUM_LABELS,

    WEIGHTS,
    WEIGHTS_SIZE,

    SPLIT_WEIGHTS,
    SPLIT_WEIGHTS_SIZE,

    NUM_BATCHES,
    BATCH,
    BATCH_SIZE
};


void split(std::vector<std::string> &splits,
           const std::string &str,
           const char delim)
{
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delim)) {
        splits.push_back(item);
    }
}

bool read_training_data(const char *file_name,
                        const char delim,
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
        split(splits, std::string(line), delim);

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
    double max_label = -1;
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
    assert(max_label != -1);

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
                                               double true_label = 1)
{
    std::vector<double> weights(feats[0].size(), 0);
    size_t count = 0;
    bool is_batch_gradient_decent = batch_size == feats.size();

    size_t weights_size = weights.size();
    size_t num_workers = std::min(weights_size, (size_t)numprocs - 1);

    // send true label to workers
    for (size_t i = 1; i <= num_workers; ++i) {
        MPI_Send(&true_label, sizeof(true_label), MPI_BYTE, i, TRUE_LABEL, MPI_COMM_WORLD);
    }

    // gradient decent
    while (data_passes != 0) {
        std::cout << "Iteration: " << count++ << std::endl;

        std::vector<std::vector<double> >::const_iterator feats_batch = feats.begin();
        std::vector<double> temp_weights(weights.size());

        // for each batch of data
        while (feats_batch < feats.end()) {
            size_t num_examples = std::min(batch_size, (size_t) (feats.end() - feats_batch));

            // send weights and batch to workers
            for (size_t i = 1; i <= num_workers; ++i) {
                // send weights
                MPI_Send(&weights_size, sizeof(weights_size), MPI_BYTE, i, WEIGHTS_SIZE, MPI_COMM_WORLD);
                MPI_Send(&weights[0], weights_size, MPI_DOUBLE, i, WEIGHTS, MPI_COMM_WORLD);

                // send batch to workers
                MPI_Send(&num_examples, sizeof(num_examples), MPI_BYTE, i, BATCH_SIZE, MPI_COMM_WORLD);
                for (size_t j = 0; j < num_examples; ++j) {
                    std::vector<double> feat = *(feats_batch + j);
                    MPI_Send(&feat[0], feat.size(), MPI_DOUBLE, i, BATCH, MPI_COMM_WORLD);
                }
            }

            // receive weights from workers
            size_t cursor = 0;
            for (size_t i = 1; i <= num_workers; ++i) {
                MPI_Status status;
                size_t split_weights_size;
                MPI_Recv(&split_weights_size, sizeof(split_weights_size), MPI_BYTE, i, SPLIT_WEIGHTS_SIZE, MPI_COMM_WORLD, &status);
                MPI_Recv(&temp_weights[cursor], split_weights_size, MPI_DOUBLE, i, SPLIT_WEIGHTS, MPI_COMM_WORLD, &status);
                cursor += split_weights_size;
            }

            assert(cursor == weights.size());

            // advance by batch size;
            feats_batch += batch_size;

            // convergence
            // TODO: workers will hang
            if (temp_weights == weights) {
                return weights;
            }

            std::copy(temp_weights.begin(), temp_weights.end(), weights.begin());
            assert(temp_weights == weights);
        }

        // shuffle the data
        if (!is_batch_gradient_decent) {
            random_shuffle(feats.begin(), feats.end());
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
    // send number of labels, batches, data passes to workers
    size_t weights_size = feats[0].size();
    size_t num_labels = label_set.size() == 2 ? 1 : label_set.size();
    size_t num_batches = feats.size() / batch_size;
    size_t num_workers = std::min(weights_size, (size_t)numprocs - 1);

    for (size_t i = 1; i <= num_workers; ++i) {
        MPI_Send(&num_labels, sizeof(num_labels), MPI_BYTE, i, NUM_LABELS, MPI_COMM_WORLD);
        MPI_Send(&num_batches, sizeof(num_batches), MPI_BYTE, i, NUM_BATCHES, MPI_COMM_WORLD);
        MPI_Send(&data_passes, sizeof(data_passes), MPI_BYTE, i, DATA_PASSES, MPI_COMM_WORLD);
    }

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

void usage()
{
    std::cout << "logistic_regression <training file> <delimiter> <learning rate> <regularization parameter> " <<
        "[<data passes (-1 for convergence)> <batch size>]" << std::endl;
}

int parameter_server(int argc, char *argv[])
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

    assert(strlen(argv[2]) == 1);
    char delimiter = argv[2][0];

    // read training data
    std::cout << "Reading training data ... ";
    read_training_data(training_file, delimiter, training, validation, label_set);
    std::cout << "[Done]" << std::endl;

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

int worker()
{
    MPI_Status status;

    // receive number of labels
    size_t num_labels;
    MPI_Recv(&num_labels, sizeof(num_labels), MPI_BYTE, 0, NUM_LABELS, MPI_COMM_WORLD, &status);

    // receive number of batches
    size_t num_batches;
    MPI_Recv(&num_batches, sizeof(num_batches), MPI_BYTE, 0, NUM_BATCHES, MPI_COMM_WORLD, &status);

    // receive data passes
    int data_passes;
    MPI_Recv(&data_passes, sizeof(data_passes), MPI_BYTE, 0, DATA_PASSES, MPI_COMM_WORLD, &status);

    while (num_labels) {
        // receive true label
        double true_label;
        MPI_Recv(&true_label, sizeof(true_label), MPI_BYTE, 0, TRUE_LABEL, MPI_COMM_WORLD, &status);

        int count_data_passes = 0;
        while (count_data_passes < data_passes) {

            size_t count_batches = 0;
            while (count_batches < num_batches) {
                // receive weights from the parameter server
                size_t weights_size;
                MPI_Recv(&weights_size, sizeof(weights_size), MPI_BYTE, 0, WEIGHTS_SIZE, MPI_COMM_WORLD, &status);
                std::vector<double> weights(weights_size);
                MPI_Recv(&weights[0], weights_size, MPI_DOUBLE, 0, WEIGHTS, MPI_COMM_WORLD, &status);

                // receive batch of features from the parameter server
                // TODO: convert to single receive or avoid sending alltogether using files directly
                size_t batch_size;
                size_t feats_size = weights_size;
                MPI_Recv(&batch_size, sizeof(batch_size), MPI_BYTE, 0, BATCH_SIZE, MPI_COMM_WORLD, &status);
                std::vector<std::vector<double> > feats_batch(batch_size, std::vector<double>(feats_size));
                for (size_t i = 0; i < batch_size; ++i) {
                    MPI_Recv(&feats_batch[i][0], feats_size, MPI_DOUBLE, 0, BATCH, MPI_COMM_WORLD, &status);
                }

                // calculate range of dimension for this worker
                const size_t num_splits = std::min(weights.size(), (size_t) numprocs - 1);
                const size_t split_size = weights.size() / num_splits;

                const size_t start = (rank - 1) * split_size;
                const size_t end = rank == numprocs - 1 ? weights.size() : start + split_size;

                // make a copy of weights
                std::vector<double> temp_weights(weights.begin() + start, weights.begin() + end);

                // calculate weights for the range
                for (size_t j = 0; j < temp_weights.size(); ++j) {
                    // calculate cost for the batch
                    double cst = cost(feats_batch.begin(), weights, batch_size, j + start, true_label);

                    // regularization is not applied to the zeroth weight
                    // TODO: can this be simplified?
                    if (j + start > 0) {
                        temp_weights[j] = weights[j + start] * (1 - learning_rate * reg_param / batch_size) - learning_rate / batch_size * cst;
                    } else {
                        temp_weights[j] = weights[j + start] - learning_rate / batch_size * cst;
                    }
                }

                // send the updated weights for the range
                size_t temp_weights_size = temp_weights.size();
                MPI_Send(&temp_weights_size, sizeof(temp_weights_size), MPI_BYTE, 0, SPLIT_WEIGHTS_SIZE, MPI_COMM_WORLD);
                MPI_Send(&temp_weights[0], temp_weights_size, MPI_DOUBLE, 0, SPLIT_WEIGHTS, MPI_COMM_WORLD);

                ++count_batches;
            }
            ++count_data_passes;
        }
        --num_labels;
    }
    return 0;
}

int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &namelen);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int ret;

    // global because used across workers
    learning_rate = std::stod(std::string(argv[3]));
    reg_param = std::stod(std::string(argv[4]));

    if (rank == 0) {
        ret = parameter_server(argc, argv);
    } else {
        ret = worker();
    }

    MPI_Finalize();
    return ret;
}
