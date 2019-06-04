#include <utils/mat_utils.h>
using namespace std;
using namespace arma;


int main() {
    umat a(2, 4, fill::randn);
    a.print();
    
    uvec idx(2);
    idx << 0 << 1;

    // Mat_Utils::set_cols(a, idx, mat(2, 2, fill::ones));
    // a.print();g

    Mat_Utils<unsigned long long>::set_rows(a, idx, umat(2, 4, fill::ones));
    a.print();

    cout << Mat_Utils<unsigned long long>::get_rows(a, idx) << endl; 
    return 0;
}
