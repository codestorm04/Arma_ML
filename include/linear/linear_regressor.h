/*
* The linear regression with least square method.
* compile with flags:
* author: Yuzhen Liu
* Date: 2019.4.1 16:55
*/

#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <armadillo>

using namespace std;
using namespace arma;


// Implements a logistic binary classifier
class Linear_Regressor {

    public:
        Linear_Regressor();
        Linear_Regressor(double penalty);  // Ridge Regression
        // ~Linear_Regressor();
        vec predict(mat x);
        void train(mat x, vec y);

    private:
        rowvec w;    // a row vector of weights
        // double b = 0; // bias
        double lr = 0.3; // Learing rate
        double penalty = 0;
};

#endif // LINEAR_REGRESSION_H