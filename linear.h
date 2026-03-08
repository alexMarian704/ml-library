#ifndef LINEAR_H
#define LINEAR_H

#import "matrix.h"

struct linear_layer {
    index_t in_features;
    index_t out_features;
    matrix* weights;
    matrix* bias;
};

linear_layer* linear_layer_create(index_t in_features, index_t out_features,
                                  uint32_t seed);
void linear_layer_destroy(linear_layer* layer);

bool linear_layer_forward(matrix* out, const matrix* input,
                          const linear_layer* layer);
scalar loss_mse(const matrix* pred, const matrix* target);
bool mse_grad(matrix* out, const matrix* pred, const matrix* target);
bool linear_layer_backward(matrix* dW, matrix* db, const matrix* input,
                           const matrix* dY);
bool update_parameters(linear_layer* layer, const matrix* dW, const matrix* db,
                       scalar learning_rate);

#endif
