#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;

int main() {
    vec a = {1, 2};
    a.print();
    // mat a = {{1}, {2}}; 
    rowvec b = {2, 3};
    b.print();
    mat res = a * b ;
    
    res.print();
    (b * a).print();

    std::vector<double> v = {1, 2, 3};
    vec c = vec({1, 2, 3});
    c.print();
    cout << c(2) << endl;

    return 0;
}
