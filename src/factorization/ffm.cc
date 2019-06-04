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

#include <factorization/ffm.h>

// Implements here
FFM::FFM() {
    // TODO: w
}


void FFM::train(mat x, vec y) {
    // TODO
}


vec FFM::predict(mat x) {
    vec res = vec(x.n_cols);
    return res;
}