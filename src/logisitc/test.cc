/*
* compile with flags:  -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.3.29 10:55
*/

#include <iostream>
#include <armadillo>
#include <cmath>
#include <logistic/logistic_classifier.h>
#include <datasets/datasets.h>

using namespace std;
using namespace arma;



int main() {
    // mat x = {{1, 1}, {2, 2}, {20, 1}, {30, 1}};
    // x = x.t();
    // vec y = {1, 1, 0, 0};

    Datasets dataset = Datasets("iris");
    vec y = dataset.y.subvec(0, 99);
    mat x = dataset.x.submat(0, 0, 3, 99);
    Logisitic_Classifier lr_classifier = Logisitic_Classifier();
    lr_classifier.train(x, y);
    int res = lr_classifier.infer(vec({6.1, 2.9, 4.7, 1.4}));
    printf("%d\n", res);
    return 0;
}