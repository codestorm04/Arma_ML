/*
* The entry of the module datasets
* compile with flags:
* author: Yuzhen Liu
* Date: 2019.3.28 11:00
*/

#ifndef DATASETS_H
#define DATASETS_H

#include <armadillo>
#include <vector>
#include <string>

using namespace std;
using namespace arma;


class Datasets {
    public:
        mat x;
        vector<string> x_names;
        vec y;
		mat y_multi;
        vector<string> y_names;
        string description;

        Datasets(string name);
        ~Datasets() {};

    private:
        void _load_iris();
        void _load_wine();
        void _load_boston();
        void _load_breast_cancer();
        void _load_diabetes();
        void _load_linnerud();
};

#endif // DATASETS_H
