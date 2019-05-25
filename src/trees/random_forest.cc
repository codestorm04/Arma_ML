/*
* Random Forest classification using C4.5 / Cart decision tree algorigthm.
* Mathmetics reference: https://blog.csdn.net/colourful_sky/article/details/82082854
                        https://blog.csdn.net/a819825294/article/details/51177435
* compile with flags: g++ logistic_classifier.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.5.18 22:49
*/

#include <trees/random_forest.h>

Random_Forest::Random_Forest(int n_trees, int max_depth, double x_fraction, double feature_fraction, double max_entropy) {
    _n_trees = n_trees;
    _max_depth = max_depth;
    _x_fraction = x_fraction;
    _feature_fraction = feature_fraction;
    _max_entropy = max_entropy; 

    vector<Tree_Classifier> trees;
    _trees.reserve(_n_trees);
    for (int i = 0; i < _n_trees; i++) {
        _trees.push_back(Tree_Classifier(max_depth));
    }
}

Random_Forest::~Random_Forest() {}


void Random_Forest::train(umat x, uvec y) {
    srand((int)time(0));
    int n_sample = x.n_cols * _x_fraction;
    uvec x_ids(n_sample);
    int n_feature = x.n_rows * _feature_fraction;

    for (int i = 0 ; i < _trees.size(); i++) {
        // do sampling with putting back
        for (int j = 0; j < n_sample; j++) {
            x_ids(j) = rand() % x.n_cols;
        }

        // do sampling without putting back
        uvec feature_ids(x.n_rows);
        for (int j = 0; j < x.n_rows; j++)
            feature_ids(j) = j;
        feature_ids = shuffle(feature_ids);
        _tree_feature_ids.push_back(feature_ids.head(n_feature));

        umat x_tmp = Mat_Utils<unsigned long long>::get_submat(x, _tree_feature_ids[i], x_ids);
        uvec y_tmp = Mat_Utils<unsigned long long>::get_rows(y, x_ids);
        _trees[i].train(x_tmp, y_tmp);
    }
}


uvec Random_Forest::predict(umat x) {
    uvec res(x.n_cols);
    for (int i = 0 ; i < x.n_cols; i++) {
        map<int, int> votes;
        for (int j = 0; j < _n_trees; j++) {
            uvec x_tmp = Mat_Utils<unsigned long long>::get_rows(umat(x.col(i)), _tree_feature_ids[j]);
            int vote_tmp = _trees[j].predict(x_tmp)(0);
            votes[vote_tmp] = votes.find(vote_tmp) == votes.end() ? 1 : votes[vote_tmp] + 1;
        }
        int max_vote = -1;
        int max_index = -1;
        for (auto it = votes.begin(); it != votes.end(); it++) {
            if (it->second > max_vote) {
                max_vote = it->second;
                max_index = it->first;
            }
        }
        res[i] = max_index;
    }
    return res;
}


