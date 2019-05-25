/*
* Classification decision trees using ID3 and C4.5 algorigthms.
* compile with flags:
* author: Yuzhen Liu
* Date: 2019.5.6 13:32
*/

#ifndef TREE_CLASSIFIER_H
#define TREE_CLASSIFIER_H

#include <armadillo>
#include <trees/treenode.h>
#include <set>
#include <map>
#include <vector>
#include <cmath>
#include <limits>
#include <float.h>

using namespace std;
using namespace arma;


// Implements classification decision tree
class Tree_Classifier {

    public:
        Tree_Classifier(int max_depth = INT_MAX, double max_entropy = DBL_MAX);
        ~Tree_Classifier();
        uvec predict(umat x);
        void train(umat x, uvec y);

    private:
        Node_Multi* _tree = NULL;
        umat _train_x;
        uvec _train_y;
        set<int> _feature_ids;
        int _max_depth;
        double _max_entropy;

        void _free_node(Node_Multi* root);
        Node_Multi* _train_helper(uvec x_ids, int depth);
        map<int, int> _get_count(uvec ids);
        int _get_label(map<int, int> label_count);
        double _entropy(map<int, int> label_count);
        pair<int, map<int, uvec>> _get_min_split_map(uvec x_ids);
        map<int, uvec> _convert_uvec(map<int, vector<int>> input);
        double _information_gain_rate(uvec x_ids, map<int, uvec> split_map);
        int _predict_helper(Node_Multi* root, uvec x);
};

#endif // TREE_CLASSIFIER_H


/* TODO:
1. pruning
2. regularization

*/