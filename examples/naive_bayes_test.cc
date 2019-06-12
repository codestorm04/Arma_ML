/*
* compile with flags:   g++ test.cc tree_classifier.cc   -std=c++14 -larmadillo -I ../../include/
* author: Yuzhen Liu
* Date: 2019.5.8 17:35
*/

#include <iostream>
#include <armadillo>
#include <cmath>
#include <stdlib.h>
#include <datasets/datasets.h>
#include <probalistic/naive_bayes.h>

using namespace std;
using namespace arma;

pair<umat, uvec> generate_data() {
    int n_sample = 100;
    int n_feature = 4;
    umat x(n_feature, n_sample);    
    uvec y(n_sample);

    // generate n samples iteratively
    srand((unsigned)time(NULL));
    for(int i =0; i < n_sample; i++) {
        x(0, i) = rand() % 10;
        x(1, i) = rand() % 10 + 100;
        x(2, i) = rand() % 10;
        x(3, i) = rand() % 10;

        if (x(0, i) <= 5 && x(1, i) > 105 && x(2, i) > 5 )  y(i) = 0;
        else if (x(0, i) > 5 && x(1, i) == 106 && x(2, i) <= 5 && x(3, i) > 5)  y(i) = 1;
        else if (x(0, i) <= 5 && x(1, i) <= 105 && x(2, i) < 5)  y(i) = 2;
        else y(i) = 3;
    }
    return make_pair(x, y);
}


int main() {
    pair<umat, uvec> p = generate_data();
    mat x = conv_to<mat>::from(p.first);
    uvec y = p.second;

    Datasets dataset = Datasets("iris");
    mat x = dataset.x;
    uvec y = conv_to<uvec>::from(dataset.y);


    // Naive_Bayes bayes_classifier = Naive_Bayes();
    // bayes_classifier.train(x, mat(0, 0), y);
    // uvec res = bayes_classifier.predict(x, mat(0, 0));

    uvec dis = res - y;
    int count =0;
    for (int i =0; i < dis.n_elem; i++) {
        if (dis(i) == 0) count++;
    }
    // dis.print();
    cout << "The accuracy is: " << (count * 100/ dis.n_elem) << "%." << endl;

    return 0;
}

