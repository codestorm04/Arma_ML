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

#include <factorization/fm.h>

// Implements here
FM::FM(double lr, int epoch, int k) : lr(lr), epoch(epoch), k(k) {}

void FM::train(mat x, uvec field, vec y) {
    // requires the size of x and field is identity
    pair<umat, mat> sparse = _to_sparse(x, field);
    umat idx = sparse.first;
    x = sparse.second;
    int n_samples = x.n_cols;

    _init(sum(field));
    // initial the w1, v, set the total feature number fot w1

    for (int round = 0; round< epoch; round++) {
        pair<vec, mat> infer_y = _get_y(idx, x, field);
        vec res = infer_y.first;
        res = res - y;
        // fot cross_entropy loss, dC_dz = a - y.  (a = _sigmoid(z))
        
        mat vv_x = infer_y.second;
        // for the pre-calculated values: sigma(v_jf, x_j)

        // w0 update
        w0 -= lr * sum(res) / n_samples;

        // w1 update
        rowvec w1_delta(size(w1), fill::zeros);
        for (int i = 0; i < n_samples; i++) {
            mat tmp = Mat_Utils<double>::get_cols(w1_delta, idx.col(i)) + x.col(i).t() * res(i);
            Mat_Utils<double>::set_cols(w1_delta, idx.col(i), tmp);
        }
        w1 -= lr * w1_delta / n_samples;

        // v update
        mat v_delta(size(v), fill::zeros);
        for (int i = 0; i < n_samples; i++) {
            // v_if*x^2:
            for (int j = 0 ; j < x.col(i).n_elem; j++) {
                v_delta.col(idx(j, i)) -= v.col(idx(j, i)) * x(j, i) * x(j, i);
            }

            // x_i * sigma(v_jf * x_j)
            mat tmp = vv_x.col(i) * x.col(i).t();
            Mat_Utils<double>::set_cols(v_delta, idx.col(i), tmp);
        }
        v -= lr * v_delta / n_samples;
    }
}


vec FM::predict(mat x, uvec field) {
    // requires the size of x and field is identity
    pair<umat, mat> sparse = _to_sparse(x, field);
    umat idx = sparse.first;
    x = sparse.second;
    return _get_y(idx, x, field).first;
}


pair<vec, mat> FM::_get_y(umat idx, mat x, uvec field) {
    // returns prediction results along with a mediate matrix used in training
    
    vec res(x.n_cols);
    mat sum_v_x(k, x.n_cols, fill::zeros);
    // used for training

    for (int i= 0; i < x.n_cols; i++) {
        vec x_sam = x.col(i);
        uvec idx_sam = idx.col(i);

        double term2 = as_scalar(Mat_Utils<double>::get_cols(w1, idx_sam) * x_sam);

        mat vv = Mat_Utils<double>::get_cols(v, idx_sam);
        sum_v_x.col(i) = vv * x_sam;        
        double term3 = (sum(square(sum_v_x.col(i))) - sum(square(vv) * square(x_sam))) / 2.0;

        res(i) = _sigmoid(w0 + term2 + term3);
    }
    return make_pair(res, sum_v_x);
}


double FM::_sigmoid(double x) {
    return 1 / (1 + exp(0 - x));
}


pair<umat, mat> FM::_to_sparse(mat x, uvec field) {
    /* requires the size of x and field is identity
       returns the indexes and feature values in sparse storage
    */
    if (x.n_rows != field.n_elem) {
        cerr << "[Error] x.col(i) and field must have the same dimensions" << endl;
        throw -1;
    }
    umat idx(size(x), fill::zeros);
    mat features(x.n_rows, x.n_cols, fill::zeros);
    int shift = 0;
    for (int i = 0; i < field.n_elem; i++) {
        if (field(i) > 1) {
            // for category features
            idx.row(i) = conv_to<umat>::from(x.row(i)) + shift;
            features.row(i).fill(1.0);
        } else if (field(i) == 1) {
            idx.row(i).fill(shift);
            features.row(i) = x.row(i);
        } else {
            cerr << "[Error] feature field size must be opsitive integer" << endl;
            throw -1;
        }
        shift += field(i);
    }
    return make_pair(idx, features);
}


void FM::_init(int feature_size) {
    w1 = rowvec(feature_size, fill::zeros);
    v = mat(k, feature_size, fill::zeros);
}