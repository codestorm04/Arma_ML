/*
* Tree node definitions for multiple and binary trees.
* compile with flags:
* author: Yuzhen Liu
* Date: 2019.5.6 13:32
*/

#ifndef TREENODE_H
#define TREENODE_H

#include <map>
using namespace std;

// for continuous feature samples
class Node_Binary {
public:
    Node_Binary(double label, int feature_id=-1, double value=0) {
        this->label = label;
        this->feature_id = feature_id;
        this->value = value;
        this->left = this->right = NULL;
    }
    int feature_id;
    double value;
    Node_Binary* left;
    Node_Binary* right;
    double label;
};

// for discrete feature samples
class Node_Multi {
public:
    Node_Multi(int feature_id=-1, int label=-1){
        this->feature_id = feature_id;
        this->label = label;
    }
    int feature_id;
    int label;
    map<int, Node_Multi*> leaves;
};

#endif // TREENODE_H





/*
 * Decision tree hierarchy


                   Dicision Tree
                    /        \
                   /          \
                  /            \
                 /              \
           discrete          continues
        (multiple tree)    (binary tree)
          +--------+        +--------+
          |ID3,C4.5|        |  cart  |
          +--------+        +--------+
           /                  |        \
          /                   |         \
         /                 classify    regression
+-----------------------+                 |
| +bagging   +Adaboost  |                 |
|   |            |      |               GBDT
|   |            |      |                 |
|   RF     tree boosting|                 |
+-----------------------+              XGBoost

*
*/