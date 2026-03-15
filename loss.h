#ifndef LOSS_H
#define LOSS_H

#import "matrix.h"

scalar loss_mse(const matrix* pred, const matrix* target);
bool mse_grad(matrix* out, const matrix* pred, const matrix* target);

scalar loss_bce(const matrix* pred, const matrix* target);
bool bce_grad(matrix* out, const matrix* pred, const matrix* target);

#endif