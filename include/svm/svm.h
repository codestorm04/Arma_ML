/*
* SVM classifier with kernel functions to handle non-linear divisable datasets.
* Mathmetics reference: 
        https://blog.csdn.net/guoziqing506/article/details/81126423
        https://blog.csdn.net/guoziqing506/article/details/81117820
        https://blog.csdn.net/guoziqing506/article/details/81119449
        https://blog.csdn.net/guoziqing506/article/details/81120354
        https://blog.csdn.net/guoziqing506/article/details/81155323
* compile with flags: g++ xxx.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.6.24 14:40
*/

#ifndef SVM_H
#define SVM_H

#include <armadillo>
#include <set>
#include <cmath>
#include <math.h>
#include <svm/kernel_funcs.h>

using namespace std;
using namespace arma;


// Implements a kernel-svm classifier
class SVM {

    public:
        SVM(double c=1.0, Kernel_Func* kn=NULL);
        ~SVM();
        vec predict(mat x);
        void train(mat x, vec y);

    private:
        // a row vector of weights, f(x) = _w*x+b, useful for linear divisable datasets,
        // for non-linear use kernels instead
        rowvec _w;
        // bias
        double _b = 0.0; 
        // lagrangian multiply coefficienct, where L(w,b,a)=f(x)+sigma(a*(1-w*x+b)), minimize L(w,b,a)
        rowvec _a;
        // loose coefficient
        double _c = 1.0;
        // threshold for a minimal changes
        double _epsilon = 0.00000001;
        // tolerrate value on alpha change
        double _toler = 0.00000001;


        // kernel function object, function val() is requeired
        Kernel_Func* _kn;
        // K_i,j cached in dataset
        mat _k_cacked;
        // y' -y cached
        rowvec _E;
        // cached ids where a = 0 and non-zero 
        set<int> _zero_a_ids, _support_vecs;

        mat _train_x;
        vec _train_y;

        void    _inital(mat x, vec y);
        void    _smo();
        int     _select_first();
        int     _select_second(int i);
        double  _clip_a(int i, int j, double a_j_new);
        void    _update_aids_b_E(int i, double a_i_new, int j, double a_j_new);
        void    _update_aids(int i, double val);
};

#endif // SVM_H