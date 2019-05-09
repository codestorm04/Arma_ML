/*
* The entry of the module datasets
* compile with flags:
* author: Yuzhen Liu
* Date: 2019.3.24 11:59
*/

#ifndef LOGISTIC_CLASSIFIER_H
#define LOGISTIC_CLASSIFIER_H

#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;


// Implements a logistic binary classifier
class Logisitic_Classifier {

    public:
        Logisitic_Classifier();
        // ~Logisitic_Classifier();
        vec predict(mat x);
        void train(mat x, vec y);
        void initialize();

    private:
        rowvec w;    // a row vector of weights
        double b = 0; // bias
        double lr = 0.5; // Learing rate
        double delta_threshhold = 10;
        int iteration = 50;

        double _logistic(vec x);
        double _logistic_derivative(vec x);
};

#endif // LOGISTIC_CLASSIFIER_H