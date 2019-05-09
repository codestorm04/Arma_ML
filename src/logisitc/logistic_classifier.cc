/*
* Mathmetics reference: https://blog.csdn.net/u014258807/article/details/80616647
* compile with flags: g++ logistic_classifier.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.3.24 11:59
*/

#include <logistic/logistic_classifier.h>

// Implements here
Logisitic_Classifier::Logisitic_Classifier() {
    // TODO: w
}


void Logisitic_Classifier::train(mat x, vec y) {
    if (x.n_cols <= 0) 
        return;
    w = rowvec(x.n_rows, fill::zeros);

    for(int round =0; round < iteration; round++) {
        rowvec delta_sum_w = rowvec(size(w), fill::zeros);
        double delta_sum_b = 0;
        for (int i = 0; i < x.n_cols; i++) {
            delta_sum_b += y(i) - Logisitic_Classifier::_logistic(x.col(i));
            delta_sum_w += delta_sum_b * (x.col(i)).t();
        }
        b += lr * as_scalar(delta_sum_b) / x.n_cols;
        w += lr * delta_sum_w / x.n_cols;
        // if (sum(delta_sum_w) >= delta_threshhold) 
        //     break;
    }
}


vec Logisitic_Classifier::predict(mat x) {
    vec res = vec(x.n_cols);
    for (int i = 0; i < x.n_cols; i++) {
        res(i) = _logistic(x.col(i)) > 0.5 ? 1 : 0;
    }
    return res;
}

double Logisitic_Classifier::_logistic(vec x) {
    return 1/(1 + exp(0 - as_scalar(w * x) - b));
}