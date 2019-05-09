/*
* The entry of the module datasets
* compile with flags:
* author: Yuzhen Liu
* Date: 2019.3.24 11:59
*/

#ifndef SOFTMAX_CLASSIFIER_H
#define SOFTMAX_CLASSIFIER_H

#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;


// Implements a logistic binary classifier
class Softmax_Classifier {

    public:
        Softmax_Classifier();
        // ~Softmax_Classifier();
        vec predict(mat x);
        void train(mat x, vec y, int n_class);

    private:
        mat w;    // a row vector of weights
        // double b = 0; // bias
        double lr = 0.1; // Learing rate
        double delta_threshhold = 10;
        int iteration = 200;
};

#endif // SOFTMAX_CLASSIFIER_H