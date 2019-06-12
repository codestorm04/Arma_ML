/*
* Introductions about this file 
* compile with flags: g++ logistic_classifier.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.4.17 13:06
*/


#include <utils/mat_utils.h>

template <typename T>
Mat<T> Mat_Utils<T>::get_cols(Mat<T> m, uvec idx) {
    Mat<T> res = Mat<T>(m.n_rows, idx.n_elem);
    for (int i = 0; i < idx.n_elem; i++) {
        res.col(i) = m.col(idx(i));
    }
    return res;
}


template <typename T>
Mat<T> Mat_Utils<T>::get_rows(Mat<T> m, uvec idx) {
    Mat<T> res = Mat<T>(idx.n_elem, m.n_cols);
    for (int i = 0; i < idx.n_elem; i++) {
        res.row(i) = m.row(idx(i));
    }
    return res;
}

template <typename T>
Mat<T> Mat_Utils<T>::get_submat(Mat<T> m, uvec idx_row, uvec idx_col) {
    Mat<T> res = Mat<T>(idx_row.n_elem, idx_col.n_elem);
    for (int i = 0; i < idx_col.n_elem; i++) {
        for (int j = 0; j < idx_row.n_elem; j++) {
            res(j, i) = m(idx_row(j), idx_col(i));
        }
    }
    return res;
}

template <typename T>
void Mat_Utils<T>::set_cols(Mat<T>& m, uvec idx, Mat<T> values) {
    if (idx.n_elem != values.n_cols || m.n_rows != values.n_rows) {    
        cerr << "[Error] dimensions of columns not match in set_cols()." <<endl;
        throw -1;
    }
    for (int i = 0; i < idx.n_elem; i++) {
        m.col(idx(i)) = values.col(i);
    }
}

template <typename T>
void Mat_Utils<T>::set_rows(Mat<T>& m, uvec idx, Mat<T> values) {
    if (idx.n_elem != values.n_rows || m.n_cols != values.n_cols) {
        cerr << "[Error] dimensions of columns not match in set_cols()." <<endl;
        throw -1;
    }
    for (int i = 0; i < idx.n_elem; i++) {
        m.row(idx(i)) = values.row(i);
    }
}

template <typename T>
void Mat_Utils<T>::set_submat(Mat<T>& m, uvec idx_row, uvec idx_col, Mat<T> values) {
    if (idx_row.n_elem != values.n_rows || idx_row.n_elem > m.n_rows
         || idx_col.n_elem != values.n_cols || idx_col.n_elem > m.n_rows) {
        cerr << "[Error] dimensions of columns not match in set_submat()." <<endl;
        throw -1;
    }
    for (int i = 0; i < idx_col.n_elem; i++) {
        for (int j = 0; j < idx_row.n_elem; j++) {
            m(idx_row(j), idx_col(i)) = values(j, i);
        }
    }
}


template mat Mat_Utils<double>::get_cols(mat, uvec);
template mat Mat_Utils<double>::get_rows(mat, uvec);
template mat Mat_Utils<double>::get_submat(mat, uvec, uvec);
template void Mat_Utils<double>::set_cols(mat&, uvec, mat);
template void Mat_Utils<double>::set_rows(mat&, uvec, mat);
template void Mat_Utils<double>::set_submat(mat&, uvec, uvec, mat);

template umat Mat_Utils<unsigned long long>::get_cols(umat, uvec);
template umat Mat_Utils<unsigned long long>::get_rows(umat, uvec);
template umat Mat_Utils<unsigned long long>::get_submat(umat, uvec, uvec);
template void Mat_Utils<unsigned long long>::set_cols(umat&, uvec, umat);
template void Mat_Utils<unsigned long long>::set_rows(umat&, uvec, umat);
template void Mat_Utils<unsigned long long>::set_submat(umat&, uvec, uvec, umat);

// int main() {
//     mat a(2, 4, fill::randn);
//     a.print();
    
//     uvec idx(2);
//     idx << 0 << 1;

//     // Mat_Utils::set_cols(a, idx, mat(2, 2, fill::ones));
//     // a.print();
//     Mat_Utils<double>::set_rows(a, idx, mat(2, 4, fill::ones));
//     a.print();
//     return 0;
// }