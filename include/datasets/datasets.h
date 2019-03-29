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
        vector<string> y_names;
        string description;

        Datasets(string name);
        ~Datasets() {};

    private:
        void _load_iris();
        void _load_wine();
};

#endif // DATASETS_H