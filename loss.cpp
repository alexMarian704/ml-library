#include "loss.h"
#include "matrix.h"
#include <algorithm>
#include <cmath>

scalar loss_mse(const matrix* pred, const matrix* target) {
    if (!matrix_same_shape(pred, target)) {
        return NAN;
    }

    scalar total = scalar(0);
    index_t size = pred->rows * pred->cols;
    for (index_t index = 0; index < size; index++) {
        scalar res = pred->data[index] - target->data[index];
        total += res * res;
    }

    return total / size;
}

bool mse_grad(matrix* out, const matrix* pred, const matrix* target) {
    if (!matrix_same_shape(out, pred) || !matrix_same_shape(out, target)) {
        return false;
    }

    index_t size = out->rows * out->cols;
    scalar scale = scalar(2) / scalar(size);
    for (index_t index = 0; index < size; index++) {
        out->data[index] = scale * (pred->data[index] - target->data[index]);
    }

    return true;
}

scalar loss_bce(const matrix* pred, const matrix* target) {
    if (!matrix_same_shape(pred, target)) {
        return NAN;
    }

    scalar total = scalar(0);
    scalar eps = scalar(0.000001);
    index_t size = pred->rows * pred->cols;
    for (index_t index = 0; index < size; index++) {
        scalar p = std::max(eps, std::min(scalar(1) - eps, pred->data[index]));
        total += target->data[index] * log(p) +
                 (scalar(1) - target->data[index]) * log(scalar(1) - p);
    }

    return -(total / size);
}

bool bce_grad(matrix* out, const matrix* pred, const matrix* target) {
    if (!matrix_same_shape(out, pred) || !matrix_same_shape(out, target)) {
        return false;
    }

    index_t size = out->rows * out->cols;
    scalar eps = scalar(0.000001);
    for (index_t index = 0; index < size; index++) {
        scalar p = std::max(eps, std::min(scalar(1) - eps, pred->data[index]));
        out->data[index] = (p - target->data[index]) / (size * p * (scalar(1) - p));
    }

    return true;
}