/*
* compile with flags:   g++ test.cc softmax_classifier.cc   ../datasets/datasets.cc -std=c++14 -larmadillo -I ../../include/
* author: Yuzhen Liu
* Date: 2019.3.29 10:55
*/

#include <iostream>
#include <armadillo>
#include <svm/svm.h>
#include <svm/kernel_funcs.h>
#include <datasets/datasets.h>

using namespace std;
using namespace arma;

int main() {
    Datasets dataset = Datasets("iris");
    mat x = dataset.x.head_cols(100);
    vec y = dataset.y.head(100);
    // negative labeled as -1
    y = y.replace(0, -1);


    // // mat x = {{0, 1}, {1, 0}, {1,1}, {-1,0}, {0,-1}, {-1,-1}};
    // mat x = {{0, 1, 1, -1, 0, -1},
    //          {1, 0, 1, 0, -1, -1}};
    // vec y = {1, 1, 1, -1, -1, -1};
    // // f(X) --> y = -x;

    SVM svm = SVM();
    svm.train(x, y);
    vec res = svm.predict(x);

    printf("The sum loss is: \n");
    vec dis = res - y;
    // dis.print();
    join_rows(res, y).print();
    // cout << "The accuracy is: " << dis / (double)x.n_elem << endl;
    
    return 0;
}

