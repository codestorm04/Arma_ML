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

#include <svm/svm.h>

// Implements here
SVM::SVM(double c, Kernel_Func* kn) {
    // bias
    double _b = 0.0; 
    // loose coefficient
    _c = c;
    // kernel function object, function val() is requeired
    if (kn != NULL)
        _kn = kn;  
    else {
        // initialize a linear kernel by default
        _kn = new Multinomial_Kernel();
    }

    // _w, _a, _train_x, _train_y, _k_cacked , _E are lazy initialized when training
}

SVM::~SVM() {
    free(_kn);
    _kn = NULL;
}

void SVM::_inital(mat x, vec y) {
    int n_samples = x.n_cols;
    int n_features = x.n_rows;

    // _w, _a, _train_x, _train_y, _k_cacked , _E, three a_ids are lazy initialized when training
    _w = rowvec(n_features, fill::zeros);
    _a = rowvec(n_samples, fill::zeros);
    _train_x = x;
    _train_y = y;

    _k_cacked = mat(n_samples, n_samples);
    for(int i =0 ;i < n_samples; i++) {
        for (int j=0; j < n_samples; j++) {
            _k_cacked(i, j) = _kn->val(_train_x.col(i), _train_x.col(j));
        }
    }
    // E=y since _w=_a=0, y'-y=0
    _E = rowvec(n_samples);
    for (int i =0; i < n_samples; i++) {
        _E(i) = sum((_a.t() % _train_y) %_k_cacked.col(i)) + _b - _train_y(i);
    }

    /*cout << "_E is: \n";
    _E.print();
    cout << "_a is: \n";
    _a.print();*/

    for (int i =0 ;i < n_samples; i++) {
        _zero_a_ids.insert(i);
        // support_vecs remains empty
    }
}

void SVM::train(mat x, vec y) {
    if (x.n_cols <= 0 || x.n_rows <= 0 && y.n_elem != x.n_cols) {
        cerr << "[Error] Input dimension error.";
        throw -1;
    }
    _inital(x, y);
    _smo();

    // when kernel is linear, calculate _w to accelarate predicting
    if (_kn->_is_linear) {
        for (auto it = _support_vecs.begin(); it != _support_vecs.end(); it++) {
            _w = _w + _a(*it) * _train_y(*it) * _train_x.col(*it).t();
        }
        _w = _w * ((Multinomial_Kernel*)_kn)->_gamma;
    }
}

void SVM::_smo() {
    while (true) {
        int i = _select_first();
        if (i < 0) {
            // no _a(i) violates kkt conditions, optimizing stops
            break;
        }
        int j = _select_second(i);
        if (j < 0) 
            break;

        double eta = _k_cacked(i,i) + _k_cacked(j,j) - 2 * _k_cacked(i, j);
        if (eta == 0)
            continue;
        double a_j_new = _a(j) + _train_y(j) * (_E(i) - _E(j)) / eta;
        a_j_new = _clip_a(i, j, a_j_new);
        // y_i*a_i + y_j*a_j = gama_const
        double a_i_new = _a(i) - (a_j_new - _a(j)) * _train_y(j) / _train_y(i);
        double diff = abs(a_i_new - _a(i)) + abs(a_j_new - _a(j));
        _update_aids_b_E(i, a_i_new, j, a_j_new);

        /*cout << "_E is: \n";   
        _E.print();
        cout << "_a is: \n";
        _a.print();*/

        _a(i) = a_i_new;
        _a(j) = a_j_new;

        if (diff <= _epsilon)
            break;
    }
}

int SVM::_select_first() {
    /*  
     * Check and pick up the alpha who violates the KKT condition
     * - satisfy KKT condition
     *     1) yi*f(i) > 1 and alpha == 0 (outside the boundary)
     *     2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
     *     3) yi*f(i) < 1 and alpha == C (between the boundary)
     * - violate KKT condition
     * because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
     *     1) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
     *     2) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
     *     3) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct) 
     * ref: https://blog.csdn.net/zouxy09/article/details/17292011 
    */

    // select first alpha a(i) from where a_i > 0
    for (auto it = _support_vecs.begin(); it != _support_vecs.end(); it++) {
        int i = *it;
        if (_train_y(i) * _E(i) < 0 && _a(i) >= _c + _toler || _train_y(i) * _E(i) != 0 && _a(i) < _c + _toler)
            return i;
    }
    // select from a_i == 0
    for (auto it = _zero_a_ids.begin(); it != _zero_a_ids.end(); it++) {
        int i = *it;
        if (_train_y(i)*_E(i) <= 0)
            return *it;
    }
    // no alpha, to stop opotimization
    return -1;
}

int SVM::_select_second(int i) {
    double E_i = _E(i), max_distance = 0;
    int res = -1;
    for (int j =0; j < _E.n_elem; j++) {
        if (abs(_E(j) - E_i) > max_distance) {
            max_distance = abs(_E(j) - E_i);
            res = j;
        }
    }
    return res;
}

double SVM::_clip_a(int i, int j, double a_j_new) {
    double L = 0.0, H = 0.0;
    if (_train_y(i) * _train_y(j) < 0) {
        L = max(0.0, _a(j) - _a(i));
        H = min(_c, _c + _a(j) - _a(i));
    } else {
        L = max(0.0, _a(j) + _a(i) - _c);
        H = min(_c, _a(j) + _c);
    }
    a_j_new = min(H, a_j_new);
    a_j_new = max(L, a_j_new);
    return a_j_new;
}

// when a_i, a_j updated, to update b, a_ids, _E and (w, when no kernel used)
void SVM::_update_aids_b_E(int i, double a_i_new, int j, double a_j_new) {
    // update a_ids sets for heuristic select in _select_first()
    _update_aids(i, a_i_new);
    _update_aids(j, a_j_new);

    // update _b
    double b1 = _b - _E(i) - _train_y(i) * _k_cacked(i, i) * (a_i_new - _a(i)) - 
                _train_y(j) * _k_cacked(j, i) * (a_j_new - _a(j));
    double b2 = _b - _E(j) - _train_y(i) * _k_cacked(i, j) * (a_i_new - _a(i)) - 
                _train_y(j) * _k_cacked(j, j) * (a_j_new - _a(j));
    double b_delta = (b1 + b2) / 2 - _b;
    _b += b_delta;

    // update _E
    _E = _E + (a_i_new - _a(i)) * _train_y(i) * _k_cacked.row(i) + (a_j_new - _a(j)) * _train_y(j) * _k_cacked.row(j) + rowvec(_E.n_elem).fill(b_delta);
}

void SVM::_update_aids(int i, double val) {
    if (_zero_a_ids.find(i) != _zero_a_ids.end() && val > 0 + _toler) {
        _zero_a_ids.erase(i);
        _support_vecs.insert(i);
    } else if (_support_vecs.find(i) != _support_vecs.end() && val <= 0 + _toler) {
        _support_vecs.erase(i);
        _zero_a_ids.insert(i);
    }
}

vec SVM::predict(mat x) {
    if (x.n_rows != _w.n_elem) {
        cerr << "[Error] Input wrong test sample feature dimensions";
        throw -1;
    }
    vec res(x.n_cols);
    for (int i =0; i < x.n_cols; i++) {
        // when kernel is linear, cached the _w
        if (_kn->_is_linear) {
            // cout << "this way by _w\n";
            res(i) = as_scalar(_w * x.col(i)) + _b > 0 ? 1 : -1;
        } else {
            // otherwise accumulate the kernel values
            double y_pred = 0.0;
            for (auto it = _support_vecs.begin(); it != _support_vecs.end(); it++) {
                y_pred += _a(*it) * _train_y(*it) * _kn->val(_train_x.col(*it), x.col(i));
            }
            res(i) = y_pred + _b > 0 ? 1 : -1;
        }
    }
    return res;
}


/*
TODO:
1. inline 
*/
