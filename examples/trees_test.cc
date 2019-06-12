/*
* compile with flags:   g++ test.cc tree_classifier.cc   -std=c++14 -larmadillo -I ../../include/
* author: Yuzhen Liu
* Date: 2019.5.8 17:35
*/

#include <iostream>
#include <armadillo>
#include <cmath>
#include <trees/treenode.h>
#include <trees/tree_classifier.h>
#include <trees/random_forest.h>
#include <trees/cart.h>
#include <trees/gradient_boosting_dt.h>
#include <stdlib.h>
#include <datasets/datasets.h>

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
    // pair<umat, uvec> p = generate_data();
    // umat x = p.first;
    // uvec y = p.second;
    // x.print();
    // y.print();    

    /*******************************
    * Tree classifier testing
    *******************************/
    // Tree_Classifier tree_classifier = Tree_Classifier(20/*max_depth=10, max_entropy=1.0*/);
    // tree_classifier.train(x, y);
    // uvec res = tree_classifier.predict(x);
    // cout << "=======================================\n";
    // res.print();
    // cout << "=======================================\n";
    // uvec dis = res - y;
    // int count = 0;
    // for (int i =0; i < dis.n_elem; i++) {
    //     count += dis(i) == 0 ? 1 : 0;
    // }
    // cout << (float)count / dis.n_elem <<endl;




    /******************************
     * random forest tesitng
    *******************************/
    // Random_Forest rf(50, 3, 0.7, 1);
    // rf.train(x, y);
    // uvec res = rf.predict(x);

    // cout << "=======================================\n";
    // res.print();
    // cout << "=======================================\n";

    // uvec dis = res - y;
    // int count = 0;
    // for (int i =0; i < dis.n_elem; i++) {
    //     count += dis(i) == 0 ? 1 : 0;
    // }
    // cout << (float)count / dis.n_elem <<endl;



    /*******************************
    * cart regression testing
    *******************************/    
    // Datasets dataset = Datasets("boston");
    // mat x = dataset.x;
    // vec y = dataset.y;
    // Cart_Regression cart(100);
    // cart.train(x, y);
    // vec res = cart.predict(x);
    // join_rows(res, y).print();
    // vec dis = res - y;
    // // dis.print();
    // cout << "The standard deviation is: " << stddev(dis) << endl;



    /*******************************
    * GBDT testing
    *******************************/    
    Datasets dataset = Datasets("boston");
    mat x = dataset.x;
    vec y = dataset.y;
    Gradient_Boosting_DT gbdt(6, 0.8, 80);
    gbdt.train(x, y);
    vec res = gbdt.predict(x);
    join_rows(res, y).print();
    vec dis = res - y;
    // dis.print();
    cout << "The standard deviation is: " << stddev(dis) << endl;

    return 0;
}

