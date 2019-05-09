/*
* Basic linear regressior and ridge regression with L2 penalty,
* Mathmetics reference: https://www.jianshu.com/p/6305f2f8077c
* compile with flags: g++ xxx.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.4.1 16:50
*/

#include <linear/linear_regressor.h>

// Implements here
Linear_Regressor::Linear_Regressor() {
    // TODO: w
}

Linear_Regressor::Linear_Regressor(double penalty) : penalty(penalty) {
    // TODO: w
}


void Linear_Regressor::train(mat x, vec y) {
    if (x.n_cols <= 0) 
        return;
    x.insert_rows(x.n_rows, rowvec(x.n_cols, fill::ones));  // insert one row with filled 0 at bottom
    mat xt = x.t();
    mat E = eye<mat>(x.n_rows, x.n_rows);
    w = y.t() * xt * (solve(x * xt + penalty * E, E));
}


vec Linear_Regressor::predict(mat x) {
    x.insert_rows(x.n_rows, rowvec(x.n_cols, fill::ones));
    vec res = vec(x.n_cols);
    for (int i = 0; i < x.n_cols; i++) {
        res(i) = as_scalar(w * x.col(i));
    }
    return res;
}