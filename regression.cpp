#include "linear.h"
#include "matrix.h"
#include "loss.h"
#include <iostream>
#include "regression.h"

int test_regression() {
    scalar rate_of_learning = scalar(0.01);
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
    matrix_set(input, 3, 0, 0);
    matrix_set(input, 3, 1, 0);
    matrix_set(input, 3, 2, 1);
    matrix_set(input, 4, 0, 1);
    matrix_set(input, 4, 1, 1);
    matrix_set(input, 4, 2, 1);
    matrix_set(input, 5, 0, 2);
    matrix_set(input, 5, 1, 1);
    matrix_set(input, 5, 2, 3);

    matrix* target = matrix_create(6, 1);
    if (!target) {
        matrix_destroy(input);
        std::cout << "Failed to allocate target" << std::endl;
        return 1;
    }

    matrix_set(target, 0, 0, 4);
    matrix_set(target, 1, 0, 6);
    matrix_set(target, 2, 0, 3);
    matrix_set(target, 3, 0, 4.5);
    matrix_set(target, 4, 0, 5.5);
    matrix_set(target, 5, 0, 8.5);

    linear_layer* layer = linear_layer_create(3, 1, 100);
    matrix* pred = matrix_create(6, 1);
    matrix* dY = matrix_create(6, 1);
    matrix* dW = matrix_create(3, 1);
    matrix* db = matrix_create(1, 1);
    int total_runs = 2000;

    for (int epoch = 0; epoch < total_runs; epoch++) {
        bool fw = linear_layer_forward(pred, input, layer);
        if (!fw) {
            std::cout << "linear_layer_forward failed" << std::endl;
            break;
        }
        if (epoch == total_runs - 1) {
            std::cout << "Pred: " << std::endl;
            matrix_print(pred);
        }
        scalar loss = loss_mse(pred, target);
        bool mse = mse_grad(dY, pred, target);
        if (!mse) {
            std::cout << "mse_grad failed" << std::endl;
            break;
        }
        if (epoch % 50 == 0) {
            std::cout << "Loss: " << loss << ", Epoch " << epoch << std::endl;
            std::cout << "Weights: " << std::endl;
            matrix_print(layer->weights);
            std::cout << "Bias: " << std::endl;
            matrix_print(layer->bias);
        }
        bool bw = linear_layer_backward(dW, db, input, dY);
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

    matrix_destroy(input);
    matrix_destroy(target);
    matrix_destroy(pred);
    matrix_destroy(dY);
    matrix_destroy(dW);
    matrix_destroy(db);
    linear_layer_destroy(layer);

    return 0;
}