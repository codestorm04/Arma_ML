/*
* compile with flags:   g++ test.cc softmax_classifier.cc   ../datasets/datasets.cc -std=c++14 -larmadillo -I ../../include/
* author: Yuzhen Liu
* Date: 2019.4.16 21:35
*/

#include <iostream>
#include <armadillo>
#include <cmath>
#include <factorization/fm.h>
#include <datasets/datasets.h>

using namespace std;
using namespace arma;



int main() {
    Datasets dataset = Datasets("iris");

    // Logistic classifier testing
    vec y = dataset.y.subvec(0, 99);
    mat x = dataset.x.submat(0, 0, 3, 99);
    uvec field(4, fill::ones);
    FM fm = FM();
    fm.train(x, field, y);
    // vec res = fm.predict(mat({{6.1, 2.9, 4.7, 1.4}, {5.1, 3.5, 1.4, 0.2}}).t());
    vec res = fm.predict(x, field);
    (res - y).print();

    return 0;
}

