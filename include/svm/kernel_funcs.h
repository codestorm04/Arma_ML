/*
* Introductions about this file 
* compile with flags: g++ xx.cc -std=c++14 -o test -larmadillo
* refs: https://www.cnblogs.com/xiaotan-code/p/6695049.html
* author: Yuzhen Liu
* Date: 2019.6.25 16:50
*/

#ifndef Kernel_Func_H
#define Kernel_Func_H

#include <armadillo>
#include <math.h>
using namespace std;
using namespace arma;


class Kernel_Func
{
    public:
        bool _is_linear = false;
        Kernel_Func() {};
        ~Kernel_Func() {};
        virtual double val(vec x1, vec x2) = 0;
};

class Multinomial_Kernel : public Kernel_Func {
    public:
        double _gamma = 1;
        double _c = 0;
        int _n = 1;

        // makes a linear kernel by default
        Multinomial_Kernel() {
            _is_linear = true;
        }

        Multinomial_Kernel(double gamma, double c, int n) {
            _gamma = gamma;
            _c = c;
            _n = n;
            if (n == 1)
                _is_linear = true;
        }
        double val(vec x1, vec x2) {
            return pow(_gamma * as_scalar(x1.t() * x2) + _c, _n);
        }
};

class Gauss_Kernel : public Kernel_Func {
    public:
        double _sigma_2 = 1;

        Gauss_Kernel() {}
        Gauss_Kernel(double sigma) {
            _sigma_2 = sigma * sigma * (-2);
        }
        double val(vec x1, vec x2) {
            return exp(as_scalar((x1 - x2).t() * (x1 - x2))  / _sigma_2);
        }
};

class Laplace_Kernel : public Kernel_Func {
    public:
        double _sigma_2 = 1;

        Laplace_Kernel() {}
        Laplace_Kernel(double sigma) {
            _sigma_2 = 0 - sigma;
        }
        double val(vec x1, vec x2) {
            return exp(sqrt(as_scalar((x1 - x2).t() * (x1 - x2))) / _sigma_2);
        }        
};

class Sigmoid_Kernel : public Kernel_Func {
    public:
        double _gamma = 1;
        double _c = 0;

        Sigmoid_Kernel() {}
        Sigmoid_Kernel(double gamma, double c) {
            _gamma = gamma;
            _c = c;
        }
        double val(vec x1, vec x2) {
            return tanh(_gamma * as_scalar(x1.t() * x2) + _c);
        }
};

#endif // Kernel_Func_H
