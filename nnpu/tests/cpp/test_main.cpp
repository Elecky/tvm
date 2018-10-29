#include <nnpusim/common/bit_packer_factory.h>
#include <memory>
#include <iostream>
#include <nnpusim/common/bit_wrapper.h>
#include <nnpusim/common/wire.h>
#include <nnpusim/sim_module.h>

using namespace nnpu;
using namespace std;

class RegisterFile : public SimModule
{
public:
    RegisterFile(WireManager &wm):
        readIdx1(wm.Get<int>("RegReadIdx1")), readIdx2(wm.Get<int>("RegReadIdx2")),
        writeIdx(wm.Get<int>("RegWriteIdx")), writeData(wm.Get<uint32_t>("RegWriteData"))
    {
        regs.resize(32);
        for (int i = 0; i != 32; ++i)
        {
            regs[i] = i;
        }
    }

    void Move() override
    {
        auto wIdx = writeIdx->Read();
        auto wData = writeData->Read();

        _updateFunc = [this, wIdx, wData]()
        {
            if (wIdx.HasData)
            {
                this->regs[wIdx.Data] = wData.Data;
            }
        };
    }

    WireData<uint32_t> Out1()
    {
        WireData<int> idx = readIdx1->Read();
        if (idx.HasData)
        {
            return WireData<uint32_t>(true, regs[idx.Data]);
        }
        else
        {
            return WireData<uint32_t>(false);
        }
    }

    const vector<uint32_t>& Values()
    {
        return regs;
    }

    void BindOutputs(WireManager &wm);

private:
    vector<uint32_t> regs;
    std::shared_ptr<WireImpl<int>> readIdx1, readIdx2, writeIdx;
    std::shared_ptr<WireImpl<uint32_t>> writeData;
};

void RegisterFile::BindOutputs(WireManager &wm)
{
    auto outWire1 = wm.Get<uint32_t>("RegOut1");
    outWire1->SubscribeWriter(Binder<uint32_t>::Bind(&RegisterFile::Out1, sharedFromBase<RegisterFile>()));
}

class ALU : public SimModule
{
public:
    ALU(WireManager &wm) : readIdx(0), calcIdx(-1), regWire(wm.Get<uint32_t>("RegOut1"))
    {
    }

    void Move() override
    {
        auto regOut = regWire->Read();
        _updateFunc = [this, regOut]()
        {
            calcIdx = readIdx;
            regData = regOut.Data;
            if (readIdx <= 31)
                readIdx = readIdx + 1;
        };
    }

    WireData<int> GetReadIdx()
    {
        return WireData<int>(true, readIdx);
    }

    WireData<int> GetWriteIdx()
    {
        if (calcIdx != -1)
            return WireData<int>(true, calcIdx);
        else
            return WireData<int>(false);
    }

    WireData<uint32_t> GetRes()
    {
        // to implement other calculation, add logic at here.
        return WireData<uint32_t>(true, regData * regData);
    }

    void BindOutputs(WireManager &wm);
private:
    int readIdx, calcIdx;
    uint32_t regData;

    std::shared_ptr<WireImpl<uint32_t>> regWire;
};

void ALU::BindOutputs(WireManager &wm)
{
    auto self = sharedFromBase<ALU>();
    wm.Get<int>("RegReadIdx1")->SubscribeWriter(
                Binder<int>::Bind(&ALU::GetReadIdx, self));
    wm.Get<int>("RegWriteIdx")->SubscribeWriter(
                Binder<int>::Bind(&ALU::GetWriteIdx, self));
    wm.Get<uint32_t>("RegWriteData")->SubscribeWriter(
                Binder<uint32_t>::Bind(&ALU::GetRes, self));
}

int main(int argc, char *(argv[]))
{
    WireManager wm;
    std::shared_ptr<RegisterFile> reg(new RegisterFile(wm));
    reg->BindOutputs(wm);

    std::shared_ptr<ALU> alu(new ALU(wm));
    alu->BindOutputs(wm);

    for (int i = 0; i <= 32; ++i)
    {
        reg->Move();
        alu->Move();

        // update
        reg->Update();
        alu->Update();
    }
    cout << endl;

    for (auto val : reg->Values())
    {
        cout << val << ' ';
    }
    cout << endl;

    return 0;
}