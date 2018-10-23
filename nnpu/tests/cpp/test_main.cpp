#include <nnpusim/common/bit_packer_factory.h>
#include <memory>
#include <iostream>
#include <nnpusim/common/bit_wrapper.h>
#include <nnpusim/common/wire.h>

using namespace nnpu;
using namespace std;

class A
{
public:
    A() : outIndex(0)
    {}

    void set_out(std::size_t index)
    {
        outIndex = index;
    }

    std::unique_ptr<int> get(int index)
    {
        if (index == outIndex)
            return unique_ptr<int>(new int(arr[index]));
        else
            return nullptr;
    }

    void set(std::size_t index, int val)
    {
        arr[index] = val;
    }

private:
    int arr[3];

    std::size_t outIndex;
};

int main(int argc, char *(argv[]))
{
    shared_ptr<A> a = std::make_shared<A>();
    a->set(0, 1);
    a->set(1, 2);
    a->set(2, 4);

    WireManager wires;
    auto wire1 = wires.Get<int>("wire1");
    wire1->SubscribeWriter(Binder<int>::Bind(&A::get, a, 0));
    wire1->SubscribeWriter(Binder<int>::Bind(&A::get, a, 1));
    wire1->SubscribeWriter(Binder<int>::Bind(&A::get, a, 2));

    auto wire2 = wires["wire1"];
    cout << *(wire2->Read<int>()) << endl;
    a->set_out(1);
    cout << *(wire2->Read<int>()) << endl;
    a->set_out(2);
    cout << *(wire2->Read<int>()) << endl;
    a->set_out(1);
    cout << *(wire2->Read<int>()) << endl;

    a.reset();

    cout << (wire2->Read<int>() == nullptr) << endl;

    //wires.Get<double>("wire1");
    
    return 0;
}