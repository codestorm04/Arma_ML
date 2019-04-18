#include <utils/mat_utils.h>
using namespace std;
using namespace arma;

int main() {
    mat a(2, 4, fill::randn);
    a.print();
    
    uvec idx(2);
    idx << 0 << 1;

    // Mat_Utils::set_cols(a, idx, mat(2, 2, fill::ones));
    // a.print();
    Mat_Utils::set_rows<double>(a, idx, mat(2, 2, fill::ones));
    a.print();
    return 0;
}
