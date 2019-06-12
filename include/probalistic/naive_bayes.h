/*
* The naive bayes classifier.
* compile with flags: 
* author: Yuzhen Liu
* Date: 2019.6.9 21:50
*/

#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include <armadillo>
#include <vector>
#include <map>
#include <utils/mat_utils.h>
#include <cmath>
#include <math.h>
#include <float.h>

using namespace std;
using namespace arma;


// Implements a naive bayes classifier
class Naive_Bayes {

    public:
        Naive_Bayes(vec P_a=vec());
        // prior probability P(A)
        
        ~Naive_Bayes();
        
        void train(mat x_continuous, mat x_discrete, uvec y);
        // handles both continuous and discrete sample features
        
        uvec predict(mat x_continuous, mat x_discrete);

    private:
        int _n_class = 0;

        vec _P_a;
        // given or calculated prior probability

        vector<vector<map<int, double> > > _P_b_a_discrete;
        // calculated P(B_i|A): class_id --> feature_id --> feature_value:P(B_i|A)

        cube _P_b_a_continuous;
        // calculated P(B_i|A): class_id --> feature_id --> gaussian parameters: N(u, sigma) 

        map<int, double> _get_x_fraction(vec x);

        map<int, uvec> _get_y_map(uvec y);

        int _predict_helper(vec x_continuous, vec x_discrete);
        

};

#endif // NAIVE_BAYES_H

