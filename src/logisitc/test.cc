/*
* compile with flags:  -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.3.29 10:55
*/

#include <iostream>
#include <armadillo>
#include <cmath>
#include <logistic/logistic_classifier.h>
#include <logistic/softmax_classifier.h>
#include <datasets/datasets.h>

using namespace std;
using namespace arma;



int main() {
    Datasets dataset = Datasets("iris");

    // Logistic classifier testing
    // vec y = dataset.y.subvec(0, 99);
    // mat x = dataset.x.submat(0, 0, 3, 99);
    // Logisitic_Classifier lr_classifier = Logisitic_Classifier();
    // lr_classifier.train(x, y);
    // vec res = lr_classifier.infer(mat({{6.1, 2.9, 4.7, 1.4}, {5.1, 3.5, 1.4, 0.2}}).t());
    // res.print();


    // softmax classifier test (for multiple classies)
    Softmax_Classifier softmax_classifier = Softmax_Classifier();
    softmax_classifier.train(dataset.x, dataset.y, 3);
    vec res = softmax_classifier.infer(dataset.x);
    // vec res = softmax_classifier.infer(mat({{6.1, 2.9, 4.7, 1.4}, {5.1, 3.5, 1.4, 0.2}, {6.1, 2.6, 5.6, 1.4}}).t());
    res.print();
    return 0;
}