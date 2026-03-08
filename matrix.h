#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>
#include <cstddef>
#include <cstdint>

using index_t = std::size_t;
using scalar = float;

struct matrix {
    index_t rows;
    index_t cols;
    scalar* data;
};

static inline index_t matrix_index(const matrix* m, index_t r, index_t c) {
    return r * m->cols + c;
}

static inline bool matrix_same_shape(const matrix* a, const matrix* b) {
    return a && b && a->data && b->data && a->rows == b->rows && a->cols == b->cols;
}

matrix* matrix_create(index_t rows, index_t cols);
void matrix_destroy(matrix* mat);

bool matrix_add(matrix* out, const matrix* a, const matrix* b);
bool matrix_subtract(matrix* out, const matrix* a, const matrix* b);
bool matrix_matmul(matrix* out, const matrix* a, const matrix* b);
bool matrix_transpose(matrix* out, const matrix* a);
bool matrix_scalar_mul(matrix* out, const matrix* a, scalar value);
bool matrix_scalar_add(matrix* out, const matrix* a, scalar value);
bool matrix_copy(matrix* out, const matrix* a);
bool matrix_sum_rows(matrix* out, const matrix* a); // out: (1, a->cols)
bool matrix_sum_cols(matrix* out, const matrix* a); // out: (a->rows, 1)
void matrix_rand_uniform(matrix* m, scalar low, scalar high, uint32_t seed); 
void matrix_rand_normal(matrix* m, scalar mean, scalar std, uint32_t seed);
bool matrix_mul_elem(matrix* out, const matrix* a, const matrix* b);
bool matrix_apply(matrix* out, const matrix* a, scalar (*fn)(scalar));
bool matrix_argmax_rows(matrix* out, const matrix* a);
bool matrix_max_rows(matrix* out, const matrix* a);
bool matrix_add_rowwise(matrix* out, const matrix* a, const matrix* b);

bool matrix_softmax_rows(matrix* out, const matrix* a);
bool matrix_relu(matrix* out, const matrix* a);
bool matrix_sigmoid(matrix* out, const matrix* a);
bool matrix_tanh(matrix* out, const matrix* a);

void matrix_xavier_uniform(matrix* m, index_t fan_in, index_t fan_out, uint32_t seed);
void matrix_xavier_normal(matrix* m, index_t fan_in, index_t fan_out, uint32_t seed);
void matrix_he_uniform(matrix* m, index_t fan_in, uint32_t seed);
void matrix_he_normal(matrix* m, index_t fan_in, uint32_t seed);

void matrix_fill(matrix* m, scalar value);
scalar matrix_get(const matrix* mat, index_t row, index_t col);
void matrix_set(matrix* mat, index_t row, index_t col, scalar value);
void matrix_print(const matrix* a);

scalar matrix_sum(const matrix* a);
scalar matrix_mean(const matrix* a);

#endif