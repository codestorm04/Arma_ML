/*
* Random Forest classification using C4.5 / Cart decision tree algorigthm.
* Mathmetics reference: https://blog.csdn.net/colourful_sky/article/details/82082854
                        https://blog.csdn.net/a819825294/article/details/51177435
* compile with flags: g++ logistic_classifier.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.5.18 22:49
*/

#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <armadillo>
#include <trees/tree_classifier.h>
#include <vector>
#include <limits>
#include <float.h>
#include <time.h>
#include <stdlib.h>
#include <utils/mat_utils.h>

using namespace std;
using namespace arma;


class Random_Forest {

    public:
        Random_Forest(int n_trees, int max_depth, double x_fraction, double feature_fraction, double max_entropy=DBL_MAX);
        ~Random_Forest();
        uvec predict(umat x);
        void train(umat x, uvec y);

    private:
        int _n_trees;
        // number of base classification DTs
        vector<Tree_Classifier> _trees;
        // base DT array
        vector<uvec> _tree_feature_ids;

        int _max_depth;
        // number of max depth of each base DTs
        double _max_entropy;
        // max entropy threshold to stop spliing of leaf nodes in base DTs
        double _x_fraction;
        // fraction of selected samples for each DT, within [0, 1]
        double _feature_fraction;
        // fraction of selected spliting features for each DT, within [0, 1]

};

#endif // RANDOM_FOREST_H

// TODO: add cart base classifier
// TODO: max_entropy threshold
// TODO: class weights