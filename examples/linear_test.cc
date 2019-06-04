/*
* compile with flags:   g++ test.cc softmax_classifier.cc   ../datasets/datasets.cc -std=c++14 -larmadillo -I ../../include/
* author: Yuzhen Liu
* Date: 2019.3.29 10:55
*/

#include <iostream>
#include <armadillo>
#include <cmath>
#include <linear/linear_regressor.h>
#include <datasets/datasets.h>

using namespace std;
using namespace arma;

int main() {
    Datasets dataset = Datasets("boston");

    // Linear_Regressor linear_regressor = Linear_Regressor(0.02);
    Linear_Regressor linear_regressor = Linear_Regressor();
    linear_regressor.train(dataset.x, dataset.y);

    vec res = linear_regressor.predict(dataset.x);
    printf("The sum loss is:\n");
    vec dis = res - dataset.y;
    // dis.print();
    join_rows(res, dataset.y).print();
    cout << "The standard deviation is: " << stddev(dis) << endl;
    
    return 0;
}

