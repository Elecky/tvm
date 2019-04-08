#include <memory>
#include <iostream>
#include <nnpusim/common/float16.h>

using namespace std;
using namespace nnpu;

int main()
{
    uint16_t raw = float16::max_val_raw();
    float16 half = float16::from_raw(raw);
    cout << static_cast<float>(half) << endl;
    return 0;
}