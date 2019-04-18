/*
* Introductions about this file 
* compile with flags: g++ logistic_classifier.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.4.17 12:54
*/

#ifndef MAT_UTILS_H
#define MAT_UTILS_H

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;


// Implements some utils for matrix operations
class Mat_Utils {

    public:
        template <typename T>
        static Mat<T> get_cols(Mat<T> m, uvec idx);

        template <typename T>
        static Mat<T> get_rows(Mat<T> m, uvec idx);

        template <typename T>
        static void set_cols(Mat<T>& m, uvec idx, Mat<T> values);

        template <typename T>
        static void set_rows(Mat<T>& m, uvec idx, Mat<T> values);

    private:


        
};

#endif // MAT_UTILS_H