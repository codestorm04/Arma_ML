/*
* The naive bayes classifier.
* References: 1. https://www.jiqizhixin.com/articles/2019-05-06-4
              2 .https://www.cnblogs.com/lesliexong/p/6907642.html
* compile with flags: 
* author: Yuzhen Liu
* Date: 2019.6.9 21:50
*/

#include <probalistic/naive_bayes.h>

/*
Bayes formular: 
    P(A|B) = (P(B|A) * P(A)) / P(B)
    P(A|B): posterior probability
    P(A):   prior probability
    P(B):   feature probobility


KEY POINTS:
1. laplace smoothing
2. implemented both for discrete and continuous features, discrete features P(B|A) use frequence; continuous featueres use Gaussian distribution
3. no need to calculate P(B) for a certain sample
4. point overflow when P(B_0|A) * P(B_1|A) * ... , use log(P(B_i|A)) instead
5. different from Max Likelihood Estimation, prior distribution P(A) in consider
6. independent same distribution of P(B_i|A)

TODO:
1. balance continuous and discrete by l
*/

// Implements here
Naive_Bayes::Naive_Bayes(vec P_a) {
    _P_a = P_a;
}

Naive_Bayes::~Naive_Bayes() {
    // TODO:
}

void Naive_Bayes::train(mat x_continuous, mat x_discrete, uvec y) {
    if (x_continuous.n_elem > 0 && x_discrete.n_elem > 0 
            && x_continuous.n_cols != x_discrete.n_cols) {
        cerr << "[Error] Must input same number of samples of continuous and discrete.";
        throw -1;
    }

    map<int, uvec> y_map = _get_y_map(y);
    if (_n_class == 0)
        _n_class = y_map.size();

    // init P(A)
    if (_P_a.n_elem != _n_class) {
        _P_a = vec(_n_class);        
        for (auto it = y_map.begin(); it != y_map.end(); it++) {
            _P_a(it->first) = (it->second).n_elem / (float)y.n_elem;
        }
    }

    // handle discrete features P(B_i|A) w.r.t. n classses
    for (int i = 0; i < _n_class; i++) {
        vector<map<int, double> > feature_value_probability;
        for (int j = 0; j < x_discrete.n_rows; j++) {
            vec feature_values = conv_to<vec>::from(Mat_Utils<double>::get_cols(mat(x_discrete.row(j)), y_map[i]));
            // rowvec shaped
            map<int, double> value_probability = _get_x_fraction(feature_values);
            feature_value_probability.push_back(value_probability);
        }
        _P_b_a_discrete.push_back(feature_value_probability);
    }

    // handle continuous features P(B_i|A) w.r.t. n classses
    _P_b_a_continuous = cube(_n_class, x_continuous.n_rows, 2);
    for (int i = 0; i < _n_class; i++) {
        for (int j = 0; j < x_continuous.n_rows; j++) {
            vec feature_values = conv_to<vec>::from(Mat_Utils<double>::get_cols(mat(x_continuous.row(j)), y_map[i]));
            // rowvec shaped
            _P_b_a_continuous(i, j, 0) = mean(feature_values);
            _P_b_a_continuous(i, j, 1) = stddev(feature_values);
        }
    }
}

map<int, double> Naive_Bayes::_get_x_fraction(vec x) {
    map<int, double> value_count;
    for (int i = 0; i < x.n_elem; i++) {
        value_count[(int)x(i)] += 1 / (double)x.n_elem;
    }
    return value_count;
}

map<int, uvec> Naive_Bayes::_get_y_map(uvec y) {
    map<int, vector<int> > value_map;
    for (int i = 0; i < y.n_elem; i++) {
        if (value_map.find(y(i)) == value_map.end()) {
            vector<int> y_ids;
            y_ids.push_back(i);
            value_map[y(i)] = y_ids;
        } else {
            value_map[y(i)].push_back(i);
        }
    }

    map<int, uvec> res;
    for (map<int, vector<int> >::iterator it = value_map.begin(); it != value_map.end(); it++) {
        res[it->first] = conv_to<uvec>::from(it->second);        
    }
    return res;
}

uvec Naive_Bayes::predict(mat x_continuous, mat x_discrete) {
    if (x_continuous.n_elem > 0 && x_discrete.n_elem > 0 && x_continuous.n_cols != x_discrete.n_cols) {
        cerr << "[Error] Must input same number of samples of continuous and discrete" << endl;
        throw -1;
    }
    if (x_continuous.n_cols > 0 && x_continuous.n_rows != _P_b_a_continuous.n_cols 
            || x_discrete.n_cols > 0 && x_discrete.n_rows != _P_b_a_discrete[0].size()) {
        cerr << "[Error] Continuos or discrete data has wrong input feature dimensions." << endl;
        throw -1;
    }

    int n_samples = max(x_continuous.n_cols, x_discrete.n_cols);
    uvec res(n_samples);
    for (int i =0; i < n_samples; i++) {
        vec x_con = x_continuous.n_cols > 0 ? x_continuous.col(i) : vec();
        vec x_dis = x_discrete.n_cols > 0 ? x_discrete.col(i) : vec();
        res(i) = _predict_helper(x_con, x_dis);;
    }
    return res;
}

int Naive_Bayes::_predict_helper(vec x_continuous, vec x_discrete) {
    const double PI = 3.141592653589793238462643383279502884L;  // 3.14159265358979323846
    int max_class = -1;
    double max_probability = 0 - DBL_MAX;
    // DBL_MIN is the minimum of positive number

    for (int i =0; i < _n_class; i++) {
        double log_P_res = 0;

        // continuous feature: sum of log(P(B_i|A)), where P(B_i|A) ~ N(u, sigma)
        for (int j = 0; j < x_continuous.n_elem; j++) {
            double u = _P_b_a_continuous(i, j, 0);
            double sigma = _P_b_a_continuous(i, j, 1);
            double P_feature_value = (1 / (sqrt(2 * PI) * sigma)) * exp((-1) * (x_continuous(j) - u) * (x_continuous(j) - u) / (2 * sigma * sigma));
            log_P_res += log(P_feature_value);
        }

        // discrete feature: sum of log(P(B_i|A)).
        for (int j =0; j < x_discrete.n_elem; j++) {
            map<int, double> value_probability = _P_b_a_discrete[i][j];
            if (value_probability.find((int)x_discrete(j)) != value_probability.end())
                log_P_res += log(value_probability[(int)x_discrete(j)]);
            else 
                log_P_res += log(DBL_MIN); 
                // value smoothing w.r.t. loged probability
        }
        if (log_P_res > max_probability) {
            max_probability = log_P_res;
            max_class = i;
        }
    }
    return max_class;
}