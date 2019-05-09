/*
* Mathmetics reference: http://www.cnblogs.com/zongfa/p/8971213.html
* compile with flags: g++ logistic_classifier.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.3.24 11:59
*/

#include <logistic/softmax_classifier.h>

// Implements here
Softmax_Classifier::Softmax_Classifier() {
    // TODO: w
}


void Softmax_Classifier::train(mat x, vec y, int n_class) {
    if (x.n_cols <= 0) 
        return;
    w = mat(n_class, x.n_rows, fill::zeros);

    for(int round = 0; round < iteration; round++) {
        mat delta_sum_w = mat(size(w), fill::zeros);
        for (int j = 0; j < x.n_cols; j++) {
            vec a_tmp = w * x.col(j);
            for (int k = 0; k < a_tmp.n_elem; k++)
                a_tmp(k) = exp(a_tmp(k));
            double sum_z = sum(a_tmp);
            a_tmp = a_tmp / sum_z; // a1, a2, a3, ...
            a_tmp(y(j))--; 

                // (a_tmp * (x.col(j)).t()).print();
                // printf("------------------------------------------\n");

            delta_sum_w += a_tmp * (x.col(j)).t();
        }
        w.print();
        printf("=============================================================\n");        
        w -= (lr * delta_sum_w) / (x.n_cols);
        // ALERT(!!): w -= delta_w, not +=, opposite to the gradient
        // if (sum(delta_sum_w) >= delta_threshhold)
        //     break;
    }
    w.print();
}


vec Softmax_Classifier::predict(mat x) {
    vec res = vec(x.n_cols);
    for (int i = 0; i < x.n_cols; i++) {
        res(i) = (w * x.col(i)).index_max();
    }
    return res;
}