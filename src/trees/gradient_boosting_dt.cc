/*
* Gradient Boosting Tree composed by regression decision trees using cart algorigthms.
* references: https://blog.csdn.net/zpalyq110/article/details/79527653
*             https://blog.csdn.net/horizonheart/article/details/78782622
              https://mp.weixin.qq.com/s?__biz=MzAxMjUyNDQ5OA==&mid=2653557219&idx=1&sn=523502247d6a7c5f0a4c795a31dc0f47&chksm=806e3f5eb719b64850348792bc8ac680ae2632eb71699779d1cb54230a278980d8c843128438&scene=21#wechat_redirect
* compile with flags:
* author: Yuzhen Liu
* Date: 2019.6.6 20:40
*/

#include <trees/gradient_boosting_dt.h>

// Inits here
Gradient_Boosting_DT::Gradient_Boosting_DT(int n_trees, double lr, int max_depth, double min_deviation) {
    _n_trees = n_trees;
    _lr = lr;
    _max_depth = max_depth;
    _min_deviation = _min_deviation; 

    vector<Cart_Regression> trees;
    _trees.reserve(_n_trees);
    for (int i = 0; i < _n_trees; i++) {
        _trees.push_back(Cart_Regression(max_depth));
    }
}

Gradient_Boosting_DT::~Gradient_Boosting_DT() {}


void Gradient_Boosting_DT::train(mat x, vec y) {
    _mean_val = mean(y);
    vec pre_val = vec(y.n_elem);
    pre_val.fill(_mean_val);
    vec y_res = y - pre_val;

    for(int i = 0; i < _n_trees; i++) {
        _trees[i].train(x, y_res);        
        pre_val += _lr * y_res;
        y_res = y - pre_val;
    }
}

vec Gradient_Boosting_DT::predict(mat x) {
    vec res(x.n_cols);
    res.fill(_mean_val);

    vec res_tmp(x.n_cols, fill::zeros);
    mat tmp(x.n_cols, _n_trees);

    for (int i = 0 ; i < _n_trees; i++) {
        tmp.col(i) = _trees[i].predict(x);
        res_tmp += tmp.col(i);
    }

    tmp.print();
    return res + _lr * res_tmp;
}
