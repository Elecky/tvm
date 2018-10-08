#include <nnpusim/bit_packer_factory.h>
#include <memory>
#include <iostream>
#include <nnpusim/bit_wrapper.h>

using namespace nnpu;
using std::unique_ptr;
using Byte = typename BitPacker::Byte;
using std::cout;
using std::endl;

int main(int argc, char *(argv[]))
{
    const int size = 32;
    auto raw = unique_ptr<Byte[]>(new Byte[size]);
    for (int i = 0; i != size / GetElementBytes(Type::Int16); ++i)
    {
        *(reinterpret_cast<int16_t*>(raw.get()) + i) = i;
    }

    BitWrapper arr(Type::Int16, size / sizeof(int16_t));
    arr.CopyFrom(raw.get(), 0, 32);
    auto arr2 = arr.Cast(Type::Float32);
    arr2 = arr2.Exp();
    
    //auto arr2 = arr->Exp();
    // print and see the result
    cout << arr2.GetNElem() << endl;
    for (int i = 0; i != arr2.GetNElem(); ++i)
    {
        int32_t value;
        arr2.GetAs(i, Type::Int32, &value);
        std::cout << value << " ";
    }
    std::cout << std::endl;
    
    arr2 = arr2 + arr;
    for (int i = 0; i != arr2.GetNElem(); ++i)
    {
        float value;
        arr2.GetAs(i, Type::Float32, &value);
        std::cout << value << " ";
    }
    std::cout << std::endl;

    cout << static_cast<std::size_t>(arr.GetType()) << std::endl;
    cout << static_cast<std::size_t>(arr2.GetType()) << std::endl;

    return 0;
}