#include "matrix.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>

matrix* matrix_create(index_t rows, index_t cols) {
    if (rows == 0 || cols == 0) {
        return nullptr;
    }

    matrix* m = (matrix*)malloc(sizeof(matrix));
    if (m == NULL) {
        return nullptr;
    }

    m->rows = rows;
    m->cols = cols;
    m->data = (scalar*)calloc(rows * cols, sizeof(scalar));
    if (m->data == NULL) {
        free(m);
        return nullptr;
    }

    return m;
}

void matrix_destroy(matrix* mat) {
    if (mat == NULL) {
        return;
    }

    free(mat->data);
    free(mat);
}

bool matrix_add(matrix* out, const matrix* a, const matrix* b) {
    if (!(matrix_same_shape(a, b) && matrix_same_shape(a, out))) {
        return false;
    }

    index_t size = a->rows * a->cols;
    for (index_t index = 0; index < size; index++) {
        out->data[index] = a->data[index] + b->data[index];
    }

    return true;
}

bool matrix_subtract(matrix* out, const matrix* a, const matrix* b) {
    if (!(matrix_same_shape(a, b) && matrix_same_shape(a, out))) {
        return false;
    }

    index_t size = a->rows * a->cols;
    for (index_t index = 0; index < size; index++) {
        out->data[index] = a->data[index] - b->data[index];
    }

    return true;
}

bool matrix_matmul(matrix* out, const matrix* a, const matrix* b) {
    if (a == NULL || b == NULL || out == NULL || a->data == NULL ||
        out->data == NULL || b->data == NULL) {
        return false;
    }
    if (a->cols != b->rows || a->rows != out->rows || b->cols != out->cols) {
        return false;
    }
    if (out == a || out == b) {
        return false;
    }

    index_t size = out->rows * out->cols;
    for (index_t index = 0; index < size; index++) {
        index_t a_row_index = index / out->cols;
        index_t b_col_index = index % out->cols;

        out->data[index] = scalar(0);
        for (index_t element_index = 0; element_index < a->cols;
             element_index++) {
            out->data[index] += a->data[a->cols * a_row_index + element_index] *
                                b->data[b->cols * element_index + b_col_index];
        }
    }

    return true;
}

bool matrix_transpose(matrix* out, const matrix* a) {
    if (a == NULL || out == NULL || a->rows != out->cols ||
        a->cols != out->rows) {
        return false;
    }
    if (out == a) {
        return false;
    }

    index_t size = a->rows * a->cols;
    for (index_t index = 0; index < size; index++) {
        index_t row = index / a->cols;
        index_t col = index % a->cols;
        scalar value = matrix_get(a, row, col);
        matrix_set(out, col, row, value);
    }

    return true;
}

bool matrix_scalar_mul(matrix* out, const matrix* a, scalar value) {
    if (!matrix_same_shape(a, out)) {
        return false;
    }

    index_t size = out->rows * out->cols;
    for (index_t index = 0; index < size; index++) {
        out->data[index] = a->data[index] * value;
    }

    return true;
}

bool matrix_scalar_add(matrix* out, const matrix* a, scalar value) {
    if (!matrix_same_shape(a, out)) {
        return false;
    }
    if (!out->data || !a->data) {
        return false;
    }

    index_t size = out->cols * out->rows;
    for (index_t index = 0; index < size; index++) {
        out->data[index] = a->data[index] + value;
    }

    return true;
}

bool matrix_sum_rows(matrix* out, const matrix* a) {
    if (out == NULL || a == NULL || out->data == NULL || a->data == NULL) {
        return false;
    }
    if (out->rows != 1 || out->cols != a->cols) {
        return false;
    }

    for (index_t index = 0; index < a->cols; index++) {
        scalar sum = scalar(0);
        for (index_t row_index = 0; row_index < a->rows; row_index++) {
            sum += a->data[row_index * a->cols + index];
        }
        out->data[index] = sum;
    }

    return true;
}

bool matrix_sum_cols(matrix* out, const matrix* a) {
    if (out == NULL || a == NULL || out->data == NULL || a->data == NULL) {
        return false;
    }
    if (out->cols != 1 || out->rows != a->rows) {
        return false;
    }

    for (index_t index = 0; index < a->rows; index++) {
        scalar sum = scalar(0);
        for (index_t col_index = 0; col_index < a->cols; col_index++) {
            sum += a->data[index * a->cols + col_index];
        }
        out->data[index] = sum;
    }

    return true;
}

void matrix_rand_uniform(matrix* m, scalar low, scalar high, uint32_t seed) {
    if (m == NULL || m->data == NULL) {
        return;
    }
    if (low > high) {
        return;
    }

    std::mt19937 randomGenerator(seed);
    std::uniform_real_distribution<scalar> dist(low, high);

    index_t size = m->rows * m->cols;
    for (index_t index = 0; index < size; index++) {
        m->data[index] = dist(randomGenerator);
    }
}

void matrix_rand_normal(matrix* m, scalar mean, scalar stddev, uint32_t seed) {
    if (m == NULL || m->data == NULL) {
        return;
    }
    if (stddev <= 0) {
        return;
    }

    std::mt19937 randomGenerator(seed);
    std::normal_distribution<scalar> dist(mean, stddev);

    index_t size = m->rows * m->cols;
    for (index_t index = 0; index < size; index++) {
        m->data[index] = dist(randomGenerator);
    }
}

bool matrix_mul_elem(matrix* out, const matrix* a, const matrix* b) {
    if (!(matrix_same_shape(a, b) && matrix_same_shape(a, out))) {
        return false;
    }

    index_t size = a->rows * a->cols;
    for (index_t index = 0; index < size; index++) {
        out->data[index] = a->data[index] * b->data[index];
    }

    return true;
}

bool matrix_apply(matrix* out, const matrix* a, scalar (*fn)(scalar)) {
    if (!matrix_same_shape(a, out) || !fn) {
        return false;
    }

    index_t size = a->rows * a->cols;
    for (index_t index = 0; index < size; index++) {
        out->data[index] = fn(a->data[index]);
    }

    return true;
}

bool matrix_argmax_rows(matrix* out, const matrix* a) {
    if (!out || !a || !a->data || !out->data) {
        return false;
    }
    if (out->rows != a->rows || out->cols != 1 || a->cols == 0) {
        return false;
    }

    for (index_t row_index = 0; row_index < out->rows; row_index++) {
        index_t index = 0;
        scalar max = matrix_get(a, row_index, 0);
        for (index_t col_index = 1; col_index < a->cols; col_index++) {
            if (a->data[row_index * a->cols + col_index] > max) {
                index = col_index;
                max = a->data[row_index * a->cols + col_index];
            }
        }
        out->data[row_index] = scalar(index);
    }

    return true;
}

bool matrix_max_rows(matrix* out, const matrix* a) {
    if (!out || !a || !a->data || !out->data) {
        return false;
    }
    if (out->rows != a->rows || out->cols != 1 || a->cols == 0) {
        return false;
    }

    for (index_t row_index = 0; row_index < out->rows; row_index++) {
        scalar max = matrix_get(a, row_index, 0);
        for (index_t col_index = 1; col_index < a->cols; col_index++) {
            if (a->data[row_index * a->cols + col_index] > max) {
                max = a->data[row_index * a->cols + col_index];
            }
        }
        out->data[row_index] = max;
    }

    return true;
}

bool matrix_add_rowwise(matrix* out, const matrix* a, const matrix* b) {
    if (!out || !out->data || !a || !a->data || !b || !b->data) {
        return false;
    }
    if (a->cols != b->cols || b->rows != 1 || a->cols != out->cols ||
        a->rows != out->rows) {
        return false;
    }

    for (index_t row_index = 0; row_index < a->rows; row_index++) {
        index_t index = row_index * a->cols;
        for (index_t col_index = 0; col_index < a->cols; col_index++) {
            out->data[index + col_index] = a->data[index + col_index] + b->data[col_index];
        }
    }

    return true;
}

bool matrix_softmax_rows(matrix* out, const matrix* a) {
    if (!matrix_same_shape(a, out)) {
        return false;
    }

    for (index_t row_index = 0; row_index < a->rows; row_index++) {
        index_t index = row_index * a->cols;
        scalar row_max = a->data[index];
        for (index_t col_index = 1; col_index < a->cols; col_index++) {
            index_t vector_index = index + col_index;
            if (a->data[vector_index] > row_max) {
                row_max = a->data[vector_index];
            }
        }

        scalar total = scalar(0);
        for (index_t col_index = 0; col_index < a->cols; col_index++) {
            index_t vector_index = index + col_index;
            out->data[vector_index] = std::exp(a->data[vector_index] - row_max);
            total += out->data[vector_index];
        }
        if (total == scalar(0) || !std::isfinite(total)) {
            return false;
        }

        scalar inv_total = scalar(1) / total;
        for (index_t col_index = 0; col_index < a->cols; col_index++) {
            out->data[index + col_index] *= inv_total;
        }
    }

    return true;
}

bool matrix_relu(matrix* out, const matrix* a) {
    if (!matrix_same_shape(a, out)) {
        return false;
    }

    index_t size = a->cols * a->rows;
    for (index_t index = 0; index < size; index++) {
        out->data[index] = std::max(scalar(0), a->data[index]);
    }

    return true;
}

bool matrix_sigmoid(matrix* out, const matrix* a) {
    if (!matrix_same_shape(a, out)) {
        return false;
    }

    index_t size = a->cols * a->rows;
    for (index_t index = 0; index < size; index++) {
        out->data[index] = scalar(1) / (scalar(1) + std::exp(-a->data[index]));
    }

    return true;
}

bool matrix_tanh(matrix* out, const matrix* a) {
    if (!matrix_same_shape(a, out)) {
        return false;
    }

    index_t size = a->rows * a->cols;
    for (index_t index = 0; index < size; index++) {
        out->data[index] = std::tanh(a->data[index]);
    }

    return true;
}

void matrix_xavier_uniform(matrix* m, index_t fan_in, index_t fan_out,
                           uint32_t seed) {
    if (!m || !m->data) {
        return;
    }
    if (fan_in == 0 || fan_out == 0) {
        return;
    }

    scalar limit = std::sqrt(scalar(6) / scalar(fan_in + fan_out));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<scalar> dist(-limit, limit);

    index_t size = m->rows * m->cols;
    for (index_t index = 0; index < size; index++) {
        m->data[index] = dist(rng);
    }
}

void matrix_xavier_normal(matrix* m, index_t fan_in, index_t fan_out,
                          uint32_t seed) {
    if (!m || !m->data) {
        return;
    }
    if (fan_in == 0 || fan_out == 0) {
        return;
    }

    scalar stddev = std::sqrt(scalar(2) / (scalar(fan_in + fan_out)));

    std::mt19937 rng(seed);
    std::normal_distribution<scalar> dist(scalar(0), stddev);

    index_t size = m->rows * m->cols;
    for (index_t index = 0; index < size; index++) {
        m->data[index] = dist(rng);
    }
}

void matrix_he_uniform(matrix* m, index_t fan_in, uint32_t seed) {
    if (!m || !m->data) {
        return;
    }
    if (fan_in == 0) {
        return;
    }

    scalar limit = std::sqrt(scalar(6) / scalar(fan_in));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<scalar> dist(-limit, limit);

    index_t size = m->rows * m->cols;
    for (index_t index = 0; index < size; index++) {
        m->data[index] = dist(rng);
    }
}
void matrix_he_normal(matrix* m, index_t fan_in, uint32_t seed) {
    if (!m || !m->data) {
        return;
    }
    if (fan_in == 0) {
        return;
    }

    scalar stddev = std::sqrt(scalar(2) / scalar(fan_in));

    std::mt19937 rng(seed);
    std::normal_distribution<scalar> dist(scalar(0), stddev);

    index_t size = m->rows * m->cols;
    for (index_t index = 0; index < size; index++) {
        m->data[index] = dist(rng);
    }
}

bool matrix_copy(matrix* out, const matrix* a) {
    if (!matrix_same_shape(a, out)) {
        return false;
    }

    index_t size = out->cols * out->rows;
    std::copy(a->data, a->data + size, out->data);

    return true;
}

void matrix_fill(matrix* m, scalar value) {
    if (m == NULL || m->data == NULL) {
        return;
    }

    index_t size = m->rows * m->cols;
    for (index_t index = 0; index < size; index++) {
        m->data[index] = value;
    }
}

scalar matrix_get(const matrix* mat, index_t row, index_t col) {
    if (!mat || !mat->data || row >= mat->rows || col >= mat->cols) {
        return NAN;
    }

    return mat->data[matrix_index(mat, row, col)];
}

void matrix_set(matrix* mat, index_t row, index_t col, scalar value) {
    if (mat == NULL || row >= mat->rows || col >= mat->cols ||
        mat->data == NULL) {
        return;
    }

    mat->data[matrix_index(mat, row, col)] = value;
}

void matrix_print(const matrix* a) {
    if (a == NULL || a->data == NULL) {
        std::cout << "NULL" << std::endl;
        return;
    }

    for (index_t row = 0; row < a->rows; row++) {
        for (index_t col = 0; col < a->cols; col++) {
            std::cout << matrix_get(a, row, col) << " ";
        }
        std::cout << std::endl;
    }
}

scalar matrix_sum(const matrix* a) {
    if (!a || !a->data) {
        return NAN;
    }

    scalar sum = 0;
    index_t size = a->rows * a->cols;
    for (index_t index = 0; index < size; index++) {
        sum += a->data[index];
    }

    return sum;
}

scalar matrix_mean(const matrix* a) {
    if (!a || !a->data || a->rows == 0 || a->cols == 0) {
        return NAN;
    }

    scalar sum = matrix_sum(a);

    return sum / (a->rows * a->cols);
}