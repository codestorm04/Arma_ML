/*
* Decision Tree using C4.5 algorithm, handling categorical features for multiple classification.
* Mathmetics reference: https://www.cnblogs.com/coder2012/p/4508602.html
                        https://shuwoom.com/?p=1452
* compile with flags: g++ logistic_classifier.cc -std=c++14 -o test -larmadillo
* author: Yuzhen Liu
* Date: 2019.5.6 13:32
*/

#include <trees/tree_classifier.h>

// Inits here
Tree_Classifier::Tree_Classifier(int max_depth, double max_entropy) {
    _max_depth = max_depth;
    _max_entropy = max_entropy;
}

Tree_Classifier::~Tree_Classifier() {
    _free_node(_tree);
}

void Tree_Classifier::_free_node(Node_Multi* root) {
    if (root == NULL) return;
    
    map<int, Node_Multi*> leaves = root->leaves;
    for(map<int, Node_Multi*>::iterator it = leaves.begin(); it != leaves.end(); it++) {
        _free_node(it->second);
    }
    free(root);
    root = NULL;
}

void Tree_Classifier::train(umat x, uvec y) {
    if (x.n_elem == 0) return;

    // init the global data used in recurrent splitting training
    _train_x = x;
    _train_y = y;

    for (int i=0; i < x.n_rows; i++)
        _feature_ids.insert(i);
    uvec x_ids(x.n_cols);
    for (int i=0; i < x.n_cols; i++)
        x_ids(i) = i;
    _tree = _train_helper(x_ids, 0);

    // release the memory during training
    _train_x = umat();
    _train_y = uvec();
}

Node_Multi* Tree_Classifier::_train_helper(uvec x_ids, int depth) {
    map<int, int> label_count = _get_count(x_ids);
    int label = _get_label(label_count);

    // splitting stop condition 
    if (depth >= _max_depth || _feature_ids.size() == 0 || _entropy(label_count) >= _max_entropy) {
        // stop spliting, marking labels
        Node_Multi* pNode = new Node_Multi(-1, label);
        return pNode;
    }

    // find the split feature that reduces entropy most
    pair<int, map<int, uvec>> pair = _get_min_split_map(x_ids);
    int max_feature_id = pair.first;
    map<int, uvec> max_split_map = pair.second;
     
    // generate the new tree node and split recursively
    Node_Multi* pNode = new Node_Multi(max_feature_id, label);
    for (map<int, uvec>::iterator it = max_split_map.begin(); it != max_split_map.end(); it++) { 
        _feature_ids.erase(max_feature_id);
        pNode->leaves[it->first] = _train_helper(it->second, depth + 1);
        _feature_ids.insert(max_feature_id);
    }
    return pNode;
}

// split and count by feature values
map<int, int> Tree_Classifier::_get_count(uvec ids) {
    map<int, int> label_count;
    for (int i=0; i < ids.n_elem; i++) {
        label_count[_train_y(ids(i))]++;
    }
    return label_count;
}

// get the node label according to the sub-dataset
int Tree_Classifier::_get_label(map<int, int> label_count) {
    int count = 0;
    int label = 0;
    for (map<int, int>::iterator it=label_count.begin(); it != label_count.end(); it++) {
        if (it->second > count)  {
            count = it->second;
            label = it->first;
        }
    }
    return label;
}

double Tree_Classifier::_entropy(map<int, int> label_count) {
    double entropy = 0.0;
    double n = 0.0;
    for(map<int, int>::iterator it = label_count.begin(); it != label_count.end(); it++) {
        n += it->second;
    }
    for(map<int, int>::iterator it = label_count.begin(); it != label_count.end(); it++) {
        double pi = (double)it->second / n;
        entropy -= pi * (log(pi) / log(2));
    }
    return entropy;
}

// split and get ids vectors map by feature value
pair<int, map<int, uvec>> Tree_Classifier::_get_min_split_map(uvec x_ids) {
    // C4.5 split feature selection
    double max_entropy_rate = DBL_MIN;
    int max_feature_id = -1;
    map<int, uvec> max_split_map;

    for (set<int>::iterator it = _feature_ids.begin(); it != _feature_ids.end(); it++) {
        int i = *it;
        map<int, vector<int>> split_map;
        for (int j = 0; j < x_ids.n_elem; j++) {
            int tmp = _train_x(i, x_ids(j));
            if (split_map.find(tmp) == split_map.end()) {
                vector<int> v(1);
                v[0] = x_ids(j);
                split_map[tmp] = v;
            } else {
                (split_map[tmp]).push_back(x_ids(j));
            }
        }
        map<int, uvec> t = _convert_uvec(split_map);
        double tmp_entropy_rate = _information_gain_rate(x_ids, t);
        if (tmp_entropy_rate > max_entropy_rate) {
            max_entropy_rate = tmp_entropy_rate;
            max_feature_id = i;
            max_split_map = t;
        }
    }
    return make_pair(max_feature_id, max_split_map);
}

map<int, uvec> Tree_Classifier::_convert_uvec(map<int, vector<int>> input) {
    map<int, uvec> result;
    for(map<int, vector<int>>::iterator it = input.begin(); it != input.end(); it++) {
        result[it->first] = conv_to<uvec>::from(it->second);
    }
    return result;
}


double Tree_Classifier::_information_gain_rate(uvec x_ids, map<int, uvec> split_map) {
    double entropy_1 = _entropy(_get_count(x_ids));
    double entropy_sum = 0.0;
    double split_information = 0.0;
    for (map<int, uvec>::iterator it = split_map.begin(); it != split_map.end(); it++) {
        double n_split = (it->second).n_elem;
        entropy_sum += (double)n_split / x_ids.n_elem * _entropy(_get_count(it->second));
        split_information -= ((double)n_split / x_ids.n_elem) * (log(n_split / x_ids.n_elem) / log(2));
    }
    return (entropy_1 - entropy_sum) / split_information;
}


uvec Tree_Classifier::predict(umat x) {
    if (_tree == NULL) {
        cerr << "[Error] null dicision tree, not classified!";
        throw -1;
    }
    uvec result(x.n_cols);
    for (int i = 0; i< x.n_cols; i++) {
        result(i) = _predict_helper(_tree, x.col(i));
    }
    return result;
}

int Tree_Classifier::_predict_helper(Node_Multi* root, uvec x) {
    if ((root->leaves).size() == 0 || 
        (root->leaves).find(x(root->feature_id)) == (root->leaves).end()) {
        return root->label;
    } else {
        return _predict_helper((root->leaves)[x(root->feature_id)], x);
    }
}
