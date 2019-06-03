/*
* Regression decision trees using cart algorigthms.
* compile with flags:
* author: Yuzhen Liu
* Date: 2019.6.2 16:21
*/

#include <trees/cart.h>

// Inits here
Cart_Regression::Cart_Regression(int max_depth, double min_deviation) {
    _max_depth = max_depth;
    _min_deviation = min_deviation;
}

Cart_Regression::~Cart_Regression() {
    _free_node(_tree);
}

void Cart_Regression::_free_node(Node_Binary* root) {
    if (root == NULL) return;
    
    _free_node(root->left);
    _free_node(root->right);
    free(root);
    root = NULL;
}

void Cart_Regression::train(mat x, vec y) {
    if (x.n_elem == 0) return;

    // init the global data used in recurrent splitting training
    _train_x = x;
    _train_y = y;

    uvec x_ids(x.n_cols);
    for (int i=0; i < x.n_cols; i++)
        x_ids(i) = i;
    _tree = _train_helper(x_ids, 0);

    // release the memory during training
    _train_x = mat();
    _train_y = vec();
}

Node_Binary* Cart_Regression::_train_helper(uvec x_ids, int depth) {
    vec y = Mat_Utils<double>::get_rows(_train_y, x_ids);
    double label = mean(y);

    // splitting stop condition 
    if (depth >= _max_depth ||  var(y) <= _min_deviation) {
        // stop spliting, marking labels
        Node_Binary* pNode = new Node_Binary(label);
        return pNode;
    }

    // find the split feature that reduces the deviation most
    pair<int, double> splition = _get_splitpoint(x_ids);
    int min_split_feature = splition.first;
    double min_split_value = splition.second;

    // split data set
    pair<uvec, uvec> splited = _do_split(x_ids, min_split_feature, min_split_value);
    uvec left_ids = splited.first;
    uvec right_ids = splited.second;

    // generate the new tree node and split recursively
    Node_Binary* pNode = new Node_Binary(label, min_split_feature, min_split_value);
    pNode->left = _train_helper(left_ids, depth + 1);
    pNode->right = _train_helper(right_ids, depth + 1);
    return pNode;
}

pair<int, double> Cart_Regression::_get_splitpoint(uvec x_ids) {
    double min_dev = DBL_MAX;
    int min_split_feature = -1;
    double min_split_value = 0;

    for (int i = 0; i < _train_x.n_rows; i++) {
        set<double> value_set;
        for (int j = 0; j < x_ids.n_elem; j++)  {
            value_set.insert(_train_x(i, x_ids(j)));
        }
        if (value_set.size() <= 1)
            continue;
        vector<double> values;
        values.assign(value_set.begin(), value_set.end());
        for (int j = 0; j < values.size() - 1; j++) {
            double split_point = (values[j] + values[j+1]) / 2.0;
            pair<uvec, uvec> tmp = _do_split(x_ids, i, split_point);
            int deviation_tmp = 0;
            deviation_tmp += var((Mat_Utils<double>::get_cols(_train_x, tmp.first)).row(i));
            deviation_tmp += var((Mat_Utils<double>::get_cols(_train_x, tmp.second)).row(i));

            if (deviation_tmp < min_dev) {
                min_dev = deviation_tmp;
                min_split_feature = i;
                min_split_value = split_point;
            }
        }
    }
    return make_pair(min_split_feature, min_split_value);
}

pair<uvec, uvec> Cart_Regression::_do_split(uvec x_ids, int split_feature, double split_value ) {
    vector<unsigned int> v1, v2;
    for (int i = 0; i < x_ids.n_elem; i++) {
        if (_train_x(split_feature, x_ids(i)) <= split_value) {
            v1.push_back(x_ids(i));
        } else {
            v2.push_back(x_ids(i));
        }
    }
    return make_pair(conv_to<uvec>::from(v1), conv_to<uvec>::from(v2));
}

vec Cart_Regression::predict(mat x) {
    if (_tree == NULL) {
        cerr << "[Error] null dicision tree, not classified!";
        throw -1;
    }
    vec result(x.n_cols);
    for (int i = 0; i< x.n_cols; i++) {
        result(i) = _predict_helper(_tree, x.col(i));
    }
    return result;
}

double Cart_Regression::_predict_helper(Node_Binary* root, vec x) {
    if (root == NULL) {
        cerr << "[Error] null dicision tree, not classified!";
        throw -1;
    }
    if (root->left == NULL || root->right == NULL) {
        return root->label;
    }
    if (x(root->feature_id) <= root->value) {
        return _predict_helper(root->left, x);
    } else {
        return _predict_helper(root->right, x);
    }
}
