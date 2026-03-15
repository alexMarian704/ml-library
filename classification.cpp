#include "classification.h"
#include "linear.h"
#include "loss.h"
#include "matrix.h"
#include <iostream>

int test_classification() {
    scalar rate_of_learning = scalar(0.02);
    matrix* input = matrix_create(6, 3);
    if (!input) {
        std::cout << "Failed to allocate input" << std::endl;
        return 1;
    }

    matrix_set(input, 0, 0, 0);
    matrix_set(input, 0, 1, 0);
    matrix_set(input, 0, 2, 0);

    matrix_set(input, 1, 0, 1);
    matrix_set(input, 1, 1, 0);
    matrix_set(input, 1, 2, 0);

    matrix_set(input, 2, 0, 0);
    matrix_set(input, 2, 1, 1);
    matrix_set(input, 2, 2, 0);

    matrix_set(input, 3, 0, 1);
    matrix_set(input, 3, 1, 1);
    matrix_set(input, 3, 2, 0);

    matrix_set(input, 4, 0, 1);
    matrix_set(input, 4, 1, 1);
    matrix_set(input, 4, 2, 1);

    matrix_set(input, 5, 0, 2);
    matrix_set(input, 5, 1, 1);
    matrix_set(input, 5, 2, 1);

    matrix* target = matrix_create(6, 1);
    if (!target) {
        matrix_destroy(input);
        std::cout << "Failed to allocate target" << std::endl;
        return 1;
    }

    matrix_set(target, 0, 0, 0);
    matrix_set(target, 1, 0, 0);
    matrix_set(target, 2, 0, 0);
    matrix_set(target, 3, 0, 0);
    matrix_set(target, 4, 0, 1);
    matrix_set(target, 5, 0, 1);

    linear_layer* layer = linear_layer_create(3, 1, 100);
    matrix* z = matrix_create(6, 1);
    matrix* pred = matrix_create(6, 1);
    matrix* dZ = matrix_create(6, 1);
    matrix* dW = matrix_create(3, 1);
    matrix* db = matrix_create(1, 1);
    int total_runs = 4000;

    for (int epoch = 0; epoch < total_runs; epoch++) {
        bool fw = linear_layer_forward(z, input, layer);
        if (!fw) {
            std::cout << "linear_layer_forward failed" << std::endl;
            break;
        }
        bool sigmoid = matrix_sigmoid(pred, z);
        if (!sigmoid) {
            std::cout << "matrix_sigmoid failed" << std::endl;
            break;
        }
        scalar loss = loss_bce(pred, target);
        index_t size = pred->rows * pred->cols;
        for (index_t index = 0; index < size; index++) {
            dZ->data[index] =
                (pred->data[index] - target->data[index]) / scalar(size);
        }

        if (epoch % 50 == 0) {
            std::cout << "Loss: " << loss << ", Epoch " << epoch << std::endl;
            std::cout << "Weights: " << std::endl;
            matrix_print(layer->weights);
            std::cout << "Bias: " << std::endl;
            matrix_print(layer->bias);
        }

        bool bw = linear_layer_backward(dW, db, input, dZ);
        if (!bw) {
            std::cout << "linear_layer_backward failed" << std::endl;
            break;
        }

        bool up = update_parameters(layer, dW, db, rate_of_learning);
        if (!up) {
            std::cout << "update_parameters failed" << std::endl;
            break;
        }
    }

    linear_layer_forward(z, input, layer);
    matrix_sigmoid(pred, z);
    std::cout << "Pred: " << std::endl;
    matrix_print(pred);

    matrix_destroy(input);
    matrix_destroy(target);
    matrix_destroy(pred);
    matrix_destroy(dZ);
    matrix_destroy(dW);
    matrix_destroy(db);
    matrix_destroy(z);
    linear_layer_destroy(layer);

    return 0;
}