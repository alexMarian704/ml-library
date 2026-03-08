#include "linear.h"
#include "matrix.h"
#include <cmath>
#include <cstdlib>

linear_layer* linear_layer_create(index_t in_features, index_t out_features,
                                  uint32_t seed) {
    if (in_features == 0 || out_features == 0) {
        return nullptr;
    }
    linear_layer* layer = (linear_layer*)malloc(sizeof(linear_layer));
    if (!layer) {
        return nullptr;
    }
    layer->in_features = in_features;
    layer->out_features = out_features;
    matrix* weights = matrix_create(in_features, out_features);
    if (weights == nullptr) {
        free(layer);
        return nullptr;
    }
    matrix* bias = matrix_create(1, out_features);
    if (bias == nullptr) {
        matrix_destroy(weights);
        free(layer);
        return nullptr;
    }

    matrix_he_uniform(weights, in_features, seed);
    matrix_fill(bias, scalar(0));
    layer->weights = weights;
    layer->bias = bias;

    return layer;
}

void linear_layer_destroy(linear_layer* layer) {
    if (!layer) {
        return;
    }
    matrix_destroy(layer->bias);
    matrix_destroy(layer->weights);
    free(layer);
}

bool linear_layer_forward(matrix* out, const matrix* input,
                          const linear_layer* layer) {
    if (!out || !input || !layer) {
        return false;
    }
    if (!out->data || !input->data || !layer->bias || !layer->weights) {
        return false;
    }
    if (out->cols != layer->out_features || input->cols != layer->in_features ||
        out->rows != input->rows) {
        return false;
    }

    bool mul_res = matrix_matmul(out, input, layer->weights);
    if (!mul_res) {
        return false;
    }
    bool add_res = matrix_add_rowwise(out, out, layer->bias);
    if (!add_res) {
        return false;
    }

    return true;
}

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

bool linear_layer_backward(matrix* dW, matrix* db, const matrix* input,
                           const matrix* dY) {
    matrix* transpose_input = matrix_create(input->cols, input->rows);

    bool trn_res = matrix_transpose(transpose_input, input);
    if (!trn_res) {
        matrix_destroy(transpose_input);
        return false;
    }
    bool mul_res = matrix_matmul(dW, transpose_input, dY);
    matrix_destroy(transpose_input);
    if (!mul_res) {
        return false;
    }
    bool row_sum = matrix_sum_rows(db, dY);
    if (!row_sum) {
        return false;
    }

    return true;
}

bool update_parameters(linear_layer* layer, const matrix* dW, const matrix* db,
                       scalar learning_rate) {
    if (!matrix_same_shape(dW, layer->weights) ||
        !matrix_same_shape(db, layer->bias)) {
        return false;
    }

    matrix* dW_lr = matrix_create(dW->rows, dW->cols);
    bool dW_res = matrix_scalar_mul(dW_lr, dW, learning_rate);
    if (!dW_res) {
        matrix_destroy(dW_lr);
        return false;
    }

    matrix* db_lr = matrix_create(db->rows, db->cols);
    bool db_res = matrix_scalar_mul(db_lr, db, learning_rate);
    if (!db_res) {
        matrix_destroy(db_lr);
        matrix_destroy(dW_lr);
        return false;
    }

    bool dW_sub_res = matrix_subtract(layer->weights, layer->weights, dW_lr);
    if (!dW_sub_res) {
        matrix_destroy(db_lr);
        matrix_destroy(dW_lr);
        return false;
    }
    bool db_sub_res = matrix_subtract(layer->bias, layer->bias, db_lr);
    matrix_destroy(db_lr);
    matrix_destroy(dW_lr);
    if (!db_sub_res) {
        return false;
    }

    return true;
}