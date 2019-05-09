/*
* Factorization Machine, mostly used in recommender systems, handling huge sparse matrixs, 
  predicting user actions such as whether click. Time complexity from O(kn) to O(knn)

* Mathmetics reference: http://www.cnblogs.com/zhangchaoyang/articles/7897085.html
                        https://www.cnblogs.com/zhangchaoyang/articles/8157893.html
                        https://blog.csdn.net/weixin_39750084/article/details/83549027
                        
* compile with flags: g++ xxx.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.4.13 23:00
*/

#ifndef FACTORIZATION_MACHINE_H
#define FACTORIZATION_MACHINE_H

#include <armadillo>
#include <cmath>
#include <utils/mat_utils.h>

using namespace std;
using namespace arma;


// Implements a ffm binary classifier
class FM {

    public:
        FM() {};
        FM(double lr, int epoch, int k);
        void train(mat x, uvec field, vec y);
        // x is the compressed samples matrix, indicating the only feature that is non-zero 
        // within one field 
        // field is the size of features for each field
        // it means that size of x.col(i) equals to field's.
        vec predict(mat x, uvec field);

    private:
        double w0 = 0;
        // bias weights
        rowvec w1;
        // w1 shaped (1, n)
        mat v;    
        // w2 = v.T * v. where v is matrix shaped (k, n), where k is the hidden vector
        // of certain attributions in x.
        int k = 5;
        double lr = 0.15; // Learing rate
        int epoch = 100;

        double _sigmoid(double x);
        double _sigmoid_d(double x);
        double _loss(double y, double label);
        
        pair<umat, mat> _to_sparse(mat x, uvec field);
        // returns 2 matrices (0: idx matrix, 1: feature matrix) to sparsify the x.
        
        void _init(int feature_size);
        // feature_size can accumulated by field
        
        pair<vec, mat> _get_y(umat idx, mat x, uvec field);
};

#endif // FACTORIZATION_MACHINE_H