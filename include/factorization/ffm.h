/*
* Factorization Machine, mostly used in recommender systems, handling huge sparse matrixs, 
  predicting user actions such as whether click. FFM (Field Factorization Machine) is a 
  little more complex than FM model with consideration on the feature field relations. And raise 
  the time complexity from O(kn) to O(knn)

* Mathmetics reference: http://www.cnblogs.com/zhangchaoyang/articles/7897085.html
                        https://www.cnblogs.com/zhangchaoyang/articles/8157893.html
                        https://blog.csdn.net/weixin_39750084/article/details/83549027
                        
* compile with flags: g++ xxx.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.4.13 23:17
*/

#ifndef FILED_FACTORIZATION_MACHINE_H
#define FILED_FACTORIZATION_MACHINE_H

#include <armadillo>

using namespace std;
using namespace arma;


// Implements a ffm binary classifier
class FFM {

    public:
        FFM();
        vec predict(mat x);
        void train(mat x, vec y);

    private:
        rowvec w;    // a row vector of weights
        double lr = 0.3; // Learing rate
        int epoch = 15;
};

#endif // FILED_FACTORIZATION_MACHINE_H