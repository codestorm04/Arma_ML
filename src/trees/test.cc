/*
* compile with flags:   g++ test.cc tree_classifier.cc   -std=c++14 -larmadillo -I ../../include/
* author: Yuzhen Liu
* Date: 2019.5.8 17:35
*/

#include <iostream>
#include <armadillo>
#include <cmath>
#include <trees/tree_classifier.h>
#include <trees/treenode.h>
#include <stdlib.h>


using namespace std;
using namespace arma;

pair<umat, uvec> generate_data() {
    int n_sample = 1000;
    int n_feature = 4;
    umat x(n_feature, n_sample);    
    uvec y(n_sample);

    // generate n samples iteratively
    srand((unsigned)time(NULL));
    for(int i =0; i < n_sample; i++) {
        x(0, i) = rand() % 10;
        x(1, i) = rand() % 10 + 100;
        x(2, i) = rand();
        x(3, i) = rand();

        if (x(0, i) <= 5 && x(1, i) > 105)  y(i) = 0;
        else if (x(0, i) > 5 && x(1, i) == 106)  y(i) = 1;
        else if (x(0, i) <= 5 && x(1, i) <= 105)  y(i) = 2;
        else y(i) = 3;
    }
    return make_pair(x, y);
}


int main() {
    pair<umat, uvec> p = generate_data();
    umat x = p.first;
    uvec y = p.second;
    x.print();
    y.print();

    Tree_Classifier tree_classifier = Tree_Classifier(/*max_depth=10, max_entropy=1.0*/);
    tree_classifier.train(x, y);
    uvec res = tree_classifier.predict(x);
    // (res - y).print();

    cout << "=======================================\n";
    res.print();
    cout << "=======================================\n";
    (res-y).print();
    return 0;
}

