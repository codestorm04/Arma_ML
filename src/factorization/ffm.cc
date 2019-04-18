/*
* Factorization Machine, mostly used in recommender systems, handling huge sparse matrixs, 
  predicting user actions such as whether click. FFM (Field Factorization Machine) is a 
  little more complex than FM model with consideration on the feature field relations. And raise 
  the time complexity from O(kn) to O(knn)

* Mathmetics reference: http://www.cnblogs.com/zhangchaoyang/articles/7897085.html
                        https://www.cnblogs.com/zhangchaoyang/articles/8157893.html
                        https://blog.csdn.net/weixin_39750084/article/details/83549027
                        
* compile with flags: g++ xxx.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.4.13 23:17
*/

#include <factorization/ffm.h>

// Implements here
FFM::FFM() {
    // TODO: w
}

FFM::FFM(double penalty) : penalty(penalty) {
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


vec Linear_Regressor::infer(mat x) {
    x.insert_rows(x.n_rows, rowvec(x.n_cols, fill::ones));
    vec res = vec(x.n_cols);
    for (int i = 0; i < x.n_cols; i++) {
        res(i) = as_scalar(w * x.col(i));
    }
    return res;
}