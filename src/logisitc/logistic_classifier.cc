/*
* Mathmetics reference: https://blog.csdn.net/u014258807/article/details/80616647
* compile with flags: g++ logistic_classifier.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.3.24 11:59
*/

#include <iostream>
#include <armadillo>
#include <cmath>
using namespace std;
using namespace arma;

class Logisitic_Classifier {

	public:
        Logisitic_Classifier();
        // ~Logisitic_Classifier();
        int infer(vec x);
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


int Logisitic_Classifier::infer(vec x) {
    return _logistic(x) > 0.5 ? 1 : 0;
}

double Logisitic_Classifier::_logistic(vec x) {
    return 1/(1 + exp(0 - as_scalar(w * x) - b));
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    mat x = {{1, 1}, {2, 2}, {20, 1}, {30, 1}};
    x = x.t();
    vec y = {1, 1, 0, 0};

    Logisitic_Classifier lr_classifier = Logisitic_Classifier();
    lr_classifier.train(x, y);
    int res = lr_classifier.infer(vec({22, 1}));
    printf("%d\n", res);
    return 0;
}
