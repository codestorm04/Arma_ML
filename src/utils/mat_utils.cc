/*
* Introductions about this file 
* compile with flags: g++ logistic_classifier.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.4.17 13:06
*/


#include <utils/mat_utils.h>

template <typename T>
Mat<T> Mat_Utils::get_cols(Mat<T> m, uvec idx) {
    Mat<T> res = Mat<T>(m.n_rows, idx.n_elem);
    for (int i = 0; i < idx.n_elem; i++) {
        res.col(i) = m.col(idx(i));
    }
    return res;
}


template <typename T>
Mat<T> Mat_Utils::get_rows(Mat<T> m, uvec idx) {
    Mat<T> res = Mat<T>(idx.n_elem, m.n_cols);
    for (int i = 0; i < idx.n_elem; i++) {
        res.row(i) = m.row(idx(i));
    }
    return res;
}


template <typename T>
void Mat_Utils::set_cols(Mat<T>& m, uvec idx, Mat<T> values) {
    if (idx.n_elem != values.n_cols || m.n_rows != values.n_rows) {    
        cerr << "[Error] dimensions of columns not match in set_cols()." <<endl;
        throw -1;
    }
    for (int i = 0; i < idx.n_elem; i++) {
        m.col(idx(i)) = values.col(i);
    }
}

template <typename T>
void Mat_Utils::set_rows(Mat<T>& m, uvec idx, Mat<T> values) {
    if (idx.n_elem != values.n_rows || m.n_cols != values.n_cols) {
        cerr << "[Error] dimensions of columns not match in set_cols()." <<endl;
        throw -1;
    }
    for (int i = 0; i < idx.n_elem; i++) {
        m.row(idx(i)) = values.row(i);
    }
}

template mat Mat_Utils::get_cols<double>(mat, uvec);
template mat Mat_Utils::get_rows<double>(mat, uvec);
template void Mat_Utils::set_cols<double>(mat&, uvec, mat);
template void Mat_Utils::set_rows<double>(mat&, uvec, mat);


// int main() {
//     mat a(2, 4, fill::randn);
//     a.print();
    
//     uvec idx(2);
//     idx << 0 << 1;

//     res.print();
//     // Mat_Utils::set_cols(a, idx, mat(2, 2, fill::ones));
//     // a.print();
//     Mat_Utils::set_rows(a, idx, mat(2, 2, fill::ones));
//     a.print();
//     return 0;
// }