/*
* Gradient Boosting Tree composed by regression decision trees using cart algorigthms.
* references: https://blog.csdn.net/zpalyq110/article/details/79527653
*             https://blog.csdn.net/horizonheart/article/details/78782622

* compile with flags:
* author: Yuzhen Liu
* Date: 2019.6.6 20:40
*/

#ifndef GRADIENT_BOOSTING_DT_H
#define GRADIENT_BOOSTING_DT_H

#include <armadillo>
#include <trees/cart.h>
#include <vector>
#include <limits>
#include <float.h>

using namespace std;
using namespace arma;


// Implements gradient boosting tree
class Gradient_Boosting_DT {

    public:
        Gradient_Boosting_DT(int n_trees, double lr=0.1, int max_depth=INT_MAX, double min_deviation=DBL_MAX);
        ~Gradient_Boosting_DT();
        vec predict(mat x);
        void train(mat x, vec y);

    private:
        int _n_trees;
        // number of base regression DTs

        vector<Cart_Regression> _trees;
        // base regression DT array

        double _mean_val;
        // the predict=mean_val + lr * sigma_1^n(_trees[i].predict())
        // averaged from y samples;

        double _lr;
        // learning rate to balance the mean_val and n trees

        int _max_depth;
        // number of max depth of each base DTs

        double _min_deviation;
        // min deviation threshold to stop spliing of leaf nodes in base regression DTs
};

#endif // GRADIENT_BOOSTING_DT_H


