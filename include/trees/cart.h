/*
* Regression decision trees using cart algorigthms.
* compile with flags:
* author: Yuzhen Liu
* Date: 2019.6.2 16:21
*/

#ifndef CART_H
#define CART_H

#include <armadillo>
#include <trees/treenode.h>
#include <utils/mat_utils.h>
#include <set>
#include <map>
#include <vector>
#include <cmath>
#include <limits>
#include <float.h>

using namespace std;
using namespace arma;


// Implements regression decision tree
class Cart_Regression {

    public:
        Cart_Regression(int max_depth = INT_MAX, double min_deviation = DBL_MIN);
        ~Cart_Regression();
        vec predict(mat x);
        void train(mat x, vec y);

    private:
        Node_Binary* _tree = NULL;
        mat _train_x;
        vec _train_y;
        int _max_depth;
        double _min_deviation;

        void _free_node(Node_Binary* root);
        Node_Binary* _train_helper(uvec x_ids, int depth);
        pair<int, double> _get_splitpoint(uvec x_ids);
        pair<uvec, uvec> _do_split(uvec x_ids, int split_feature, double split_value);
        double _predict_helper(Node_Binary* root, vec x);
};

#endif // CART_H


