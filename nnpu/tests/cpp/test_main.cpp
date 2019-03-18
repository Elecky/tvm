#include <memory>
#include <iostream>
#include <nnpusim/insn_wrapper.h>
#include <nnpusim/sc_sim/ifetch.h>
#include <nnpusim/sc_sim/idecode.h>
#include <nnpusim/sc_sim/issue_queue.h>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <nnpusim/sc_sim/future_file.h>
#include <nnpusim/sc_sim/reserve_station.h>
#include <nnpusim/sc_sim/alu.h>
#include <nnpusim/sc_sim/common_data_bus.h>
#include <nnpusim/sc_sim/branch_unit.h>
#include <nnpusim/sc_sim/load_store_unit.h>
#include <nnpusim/sc_sim/scalar_memory.h>
#include <nnpusim/sc_sim/memory_queue.h>
#include <nnpusim/sc_sim/address_generate_unit.h>
#include <nnpusim/sc_sim/vector_unit.h>

using namespace nnpu;
using namespace nnpu::sc_sim;
using namespace std;

std::vector<NNPUInsn> init_simple_insn()
{
    vector<NNPUInsn> insns;
    using Imm = nnpu::ALURegImmInsn;
    using Bin = nnpu::ALUBinaryInsn;
    using Store = nnpu::SclrStoreInsn;
    using Load = nnpu::SclrLoadInsn;

    auto AddIU = [](regNo_t Rd, regNo_t Rs, reg_t imm) -> nnpu::ALURegImmInsn
        { return nnpu::ALURegImmInsn(Rd, Rs, imm, ALURegImmOp::AddIU); };

    insns.emplace_back(AddIU(2, 0, 1));
    insns.emplace_back(AddIU(3, 2, 2));
    insns.emplace_back(AddIU(4, 2, 3));
    insns.emplace_back(AddIU(5, 2, 4));
    insns.emplace_back(AddIU(6, 2, 5));
    insns.emplace_back(AddIU(7, 2, 6));
    insns.emplace_back(nnpu::JumpInsn(0));

    cout << "Instructions: \n";
    InsnDumper dumper;
    for (auto &item : insns)
    {
        item.Call(dumper, cout);
        cout << endl;
    }
    cout << endl;

    return insns;
}

std::vector<NNPUInsn> init_insn()
{
    vector<NNPUInsn> insns;
    using Li = nnpu::LiInsn;
    using Bin = nnpu::ALUBinaryInsn;
    using Store = nnpu::SclrStoreInsn;
    using Load = nnpu::SclrLoadInsn;

    auto AddIU = [](regNo_t Rd, regNo_t Rs, reg_t imm) -> nnpu::ALURegImmInsn
        { return nnpu::ALURegImmInsn(Rd, Rs, imm, ALURegImmOp::AddIU); };
    
    insns.emplace_back(AddIU(1, 0, 5));
    insns.emplace_back(AddIU(2, 0, -1));
    insns.emplace_back(nnpu::BEZInsn(4, 1));
    insns.emplace_back(Bin(3, 1, 3, ALUBinaryOp::Add));
    insns.emplace_back(Bin(1, 1, 2, ALUBinaryOp::Add));
    insns.emplace_back(nnpu::JumpInsn(-3));
    insns.emplace_back(nnpu::JumpInsn(0));

    InsnDumper dumper;
    for (auto &item : insns)
    {
        item.Call(dumper, cout);
        cout << endl;
    }

    return insns;
}

std::vector<NNPUInsn> load_store_test_insns()
{
    vector<NNPUInsn> insns;
    using Bin = nnpu::ALUBinaryInsn;
    using Store = nnpu::SclrStoreInsn;
    using Load = nnpu::SclrLoadInsn;
    using Unary = nnpu::ALURegImmInsn;

    auto Li = [](regNo_t rd, reg_t imm) 
        { return Unary(rd, 0, imm, ALURegImmOp::AddIU); };

    insns.emplace_back(Li(0, 0));
    insns.emplace_back(Li(16, 7));
    insns.emplace_back(Li(1, 1));
    insns.emplace_back(Store(1, 0, 0));
    insns.emplace_back(Li(1, 2));
    insns.emplace_back(Store(1, 0, 4));
    insns.emplace_back(Li(1, 3));
    insns.emplace_back(Store(1, 0, 8));
    insns.emplace_back(Li(1, 65537));
    insns.emplace_back(Store(1, 0, 12));
    
    insns.emplace_back(Li(1, 4));  // $0 <- i, value = #4
    //insns.emplace_back(Li(4, 4));  // $4 == 4
    //insns.emplace_back(nnpu::BEZInsn(7, 0));
    
    insns.emplace_back(Unary(2, 1, 4, ALURegImmOp::MulIU));  // $2 <- 4*i
    insns.emplace_back(Load(3, 2, -4));  // $3 <- load $2 - 4
    insns.emplace_back(Bin(3, 16, 3, ALUBinaryOp::Add));
    insns.emplace_back(Unary(1, 1, -1, ALURegImmOp::AddIU));  // i = i - 1
    insns.emplace_back(Store(3, 2, 12));

    insns.emplace_back(nnpu::BNEZInsn(-5, 1));
    //insns.emplace_back(nnpu::JumpInsn(-6));
    insns.emplace_back(Load(31, 0, 28));
    insns.emplace_back(nnpu::JumpInsn(0));

    InsnDumper dumper;
    for (auto &item : insns)
    {
        item.Call(dumper, cout);
        cout << endl;
    }

    return insns;
}

std::vector<NNPUInsn> insert_sort_insns()
{
    vector<NNPUInsn> insns;
    using Bin = nnpu::ALUBinaryInsn;
    using Store = nnpu::SclrStoreInsn;
    using Load = nnpu::SclrLoadInsn;
    using Unary = nnpu::ALURegImmInsn;

    auto Li = [](regNo_t rd, reg_t imm) 
        { return Unary(rd, 0, imm, ALURegImmOp::AddIU); };

    // insns.emplace_back(Li(0, 0));
    insns.emplace_back(Li(1, 29));
    insns.emplace_back(Store(1, 0, 0));
    insns.emplace_back(Li(1, 255));
    insns.emplace_back(Store(1, 0, 4));
    insns.emplace_back(Li(1, 8));
    insns.emplace_back(Store(1, 0, 8));
    insns.emplace_back(Li(1, 65537));
    insns.emplace_back(Store(1, 0, 12));
    insns.emplace_back(Li(1, 233));
    insns.emplace_back(Store(1, 0, 16));

    // insert sort
    insns.emplace_back(Li(1, 1));  // let $1=i;  $1 <= 1

    insns.emplace_back(Unary(2, 1, 5, ALURegImmOp::SLTIU));  // $2 = $1 < 5
    insns.emplace_back(BEZInsn(13, 2));  // BEZ $2, #??

    insns.emplace_back(Unary(3, 1, 4, ALURegImmOp::MulIU));  // let $3=j <= 4 * i
    insns.emplace_back(Load(4, 3, 0));  // let $4 = arr[i]

    insns.emplace_back(BEZInsn(7, 3));  // if j == 0, end while
    insns.emplace_back(Load(5, 3, -4));  // load a[j / 4 - 1]
    insns.emplace_back(Bin(2, 4, 5, ALUBinaryOp::SLTU));  // key < a[j / 4 - 1] ?
    insns.emplace_back(BEZInsn(4, 2));  // if not, end while
    // if key is less
    insns.emplace_back(Store(5, 3, 0));  // a[j / 4] = a[j / 4 - 1]
    insns.emplace_back(Unary(3, 3, -4, ALURegImmOp::AddIU));  // j = j - 4
    insns.emplace_back(JumpInsn(-6));

    // end of while
    insns.emplace_back(Store(4, 3, 0));


    insns.emplace_back(Unary(1, 1, 1, ALURegImmOp::AddIU));
    insns.emplace_back(JumpInsn(-13));

    // load all for once
    insns.emplace_back(Li(1, 0));  // let $1=i;  $1 <= 0

    insns.emplace_back(Unary(2, 1, 20, ALURegImmOp::SLTIU));  // $2 = $1 < 20
    insns.emplace_back(BEZInsn(4, 2));
    insns.emplace_back(Load(16, 1, 0));
    insns.emplace_back(Unary(1, 1, 4, ALURegImmOp::AddIU));
    insns.emplace_back(JumpInsn(-4));

    insns.emplace_back(JumpInsn(0));

    InsnDumper dumper;
    for (auto &item : insns)
    {
        item.Call(dumper, cout);
        cout << endl;
    }

    return insns;
}

std::vector<NNPUInsn> vctr_test_insns()
{
    vector<NNPUInsn> insns;
    using Binary = nnpu::ALUBinaryInsn;
    using Store = nnpu::SclrStoreInsn;
    using Load = nnpu::SclrLoadInsn;
    using Unary = nnpu::ALURegImmInsn;

    auto Li = [](regNo_t rd, reg_t imm) 
        { return Unary(rd, 0, imm, ALURegImmOp::AddIU); };
    
    /* loop and add:
    insns.push_back(Li(1, 0));
    insns.push_back(Li(3, 32));
    insns.push_back(Unary(2, 1, 4, ALURegImmOp::SLTIU));
    insns.push_back(BEZInsn(5, 2));

    insns.push_back(Unary(4, 1, 3, ALURegImmOp::SHLI));
    insns.push_back(Unary(1, 1, 1, ALURegImmOp::AddIU));
    insns.push_back(VctrBinaryInsn(VctrBinaryOp::Add, 4, 3, 4, 8, ModeCode::N));

    insns.push_back(JumpInsn(-5));
    
    insns.push_back(JumpInsn(0));*/

    insns.push_back(Li(3, 32));
    insns.push_back(Li(4, 0));
    insns.push_back(VctrBinaryInsn(VctrBinaryOp::Add, 4, 3, 4, 8, ModeCode::N));
    insns.push_back(Li(4, 8));
    insns.push_back(VctrBinaryInsn(VctrBinaryOp::Add, 4, 3, 4, 8, ModeCode::N));
    insns.push_back(Li(4, 16));
    insns.push_back(VctrBinaryInsn(VctrBinaryOp::Add, 4, 3, 4, 8, ModeCode::N));
    insns.push_back(Li(4, 24));
    insns.push_back(VctrBinaryInsn(VctrBinaryOp::Add, 4, 3, 4, 8, ModeCode::N));
    insns.push_back(JumpInsn(0));

    /*
    insns.push_back(Li(0, 0));
    insns.push_back(Li(1, 40));
    insns.push_back(BufferLSInsn(LSDIR::Load, 0, 0, 1));

    insns.push_back(Li(3, 32));
    insns.push_back(Li(4, 0));
    insns.push_back(VctrBinaryInsn(VctrBinaryOp::Add, 4, 3, 4, 8, ModeCode::N));
    insns.push_back(Li(4, 8));
    insns.push_back(VctrBinaryInsn(VctrBinaryOp::Add, 4, 3, 4, 8, ModeCode::N));
    insns.push_back(Li(4, 16));
    insns.push_back(VctrBinaryInsn(VctrBinaryOp::Add, 4, 3, 4, 8, ModeCode::N));
    insns.push_back(Li(4, 24));
    insns.push_back(Li(5, 8));
    insns.push_back(Li(1, 1));
    insns.push_back(MemsetInsn(4, 5, 1, ModeCode::N, 100));
    insns.push_back(VctrBinaryInsn(VctrBinaryOp::Add, 4, 3, 4, 8, ModeCode::N));

    insns.push_back(Li(1, 32));
    insns.push_back(BufferLSInsn(LSDIR::Store, 0, 0, 1));*/

    InsnDumper dumper;
    for (auto &item : insns)
    {
        item.Call(dumper, cout);
        cout << endl;
    }

    return insns;
}

class Monitor : public sc_module
{
public:
    sc_in<bool> clk;
    sc_vector<sc_in<InsnWrapper>> insns;

    SC_HAS_PROCESS(Monitor);
    Monitor(sc_module_name name, std::size_t _depth) : 
        sc_module(name),
        insns("monitor_input_instructions"),
        depth(_depth)
    {
        insns.init(depth);

        SC_METHOD(Proc);
        sensitive << clk.pos();
    }

    void Proc()
    {
        std::cout << "Monitor @" << sc_time_stamp() << ":\n";
        static InsnDumper dumper;
        for (std::size_t idx = 0; idx < depth; ++idx)
        {
            insns.at(idx).read().Insn().Call(dumper, cout);
            cout << endl;
        }
        cout << endl;
    }

private:
    std::size_t depth;
};

/*
void test_memory_queue()
{
    using Unary = nnpu::ALURegImmInsn;

    auto Li = [](regNo_t rd, reg_t imm) 
        { return Unary(rd, 0, imm, ALURegImmOp::AddIU); };
    
    memory_queue mq("memory-queue", 8);

    std::vector<NNPUInsn> insns { Li(1, 1), Li(2, 2), Li(3, 3), Li(4, 4) };
    InsnDumper dumper;

    InsnWrapper insn;
    std::unordered_set<issue_id_t> empty;
    std::vector<AddressRange> noRange;

    insn = InsnWrapper(&insns[0], 0);
    insn.SetIssueId(0);
    assert(mq.nb_push(insn, noRange, noRange, empty));
    sc_start(5, SC_NS);

    insn = InsnWrapper(&insns[1], 1);
    insn.SetIssueId(1);
    assert(mq.nb_push(insn, noRange, noRange, {0}));
    sc_start(5, SC_NS);

    assert(mq.nb_read(insn, UnitType::Matrix));
    sc_start(5, SC_NS);
    insn.Insn().Call(dumper, std::cout);
    std::cout << "\n";

    assert(!mq.nb_read(insn, UnitType::Matrix));
    sc_start(5, SC_NS);

    insn = InsnWrapper(&insns[2], 2);
    insn.SetIssueId(2);
    assert(mq.nb_push(insn, noRange, noRange, {0}));
    mq.commit(0);
    mq.retire(0);
    assert(!mq.nb_read(insn, UnitType::Matrix));
    sc_start(5, SC_NS);

    assert(mq.nb_read(insn, UnitType::Matrix));
    sc_start(5, SC_NS);
    insn.Insn().Call(dumper, std::cout);
    std::cout << "\n";

    mq.commit(1);
    mq.retire(1);
    insn = InsnWrapper(&insns[3], 3);
    insn.SetIssueId(3);
    assert(mq.nb_push(insn, noRange, noRange, {1}));
    assert(mq.nb_read(insn, UnitType::Matrix));
    sc_start(5, SC_NS);
    insn.Insn().Call(dumper, std::cout);
    std::cout << "\n";

    mq.commit(2);
    mq.retire(2);
    assert(mq.nb_read(insn, UnitType::Matrix));
    sc_start(5, SC_NS);
    insn.Insn().Call(dumper, std::cout);
    std::cout << "\n";

    mq.print();
    mq.commit(3);
    mq.retire(3);
    std::cout << std::endl;
    mq.print();
    sc_start(5, SC_NS);

    std::cout << std::endl;
    mq.print();
}*/

extern "C" int sc_main( int argc, char* argv[] );

int sc_main(int argc, char* argv[])
{
    YAML::Node cfg = YAML::LoadFile("/home/jian/repositories/tvm/nnpu/nnpu_config.yaml");

    const std::size_t issueDepth {cfg["issue_queue"]["issue_depth"].as<size_t>()};
    
    sc_clock clk("clock", 5, SC_NS);

    sc_signal<bool> jump_valid("jump_valid");
    sc_signal<reg_t> jump_address("jump_address");
    sc_signal<bool> branch_valid("branch_valid");
    sc_signal<reg_t> branch_address("branch_address");
    sc_signal<bool> branch_known("branch_known");
    sc_signal<bool> stall("stall");
    sc_vector<sc_signal<InsnWrapper>> IF_ID("IF->ID", issueDepth);
    sc_vector<sc_signal<InsnWrapper>> ID_IQ("ID->IQ", issueDepth);

    future_file reg_file("future_file", cfg);
    common_data_bus cdb("common_data_bus");
    random_reserve_station alu_RS("alu_reserve_station", 8);
    random_reserve_station bu_RS("branch_reserve_staion", 1);
    sequential_reserve_station ls_RS("load/store_reserve_station", 4);
    sequential_reserve_station tensor_RS("tensor_reserve_station", 4);

    // memory queues
    memory_queue vector_mq("vector-memory-queue", 4);
    memory_queue matrix_mq("matrix-memory-queue", 4);

    retire_bus r_bus("retire-bus");
    r_bus.memory_queues.bind(vector_mq);
    r_bus.memory_queues.bind(matrix_mq);

    cdb.future_file_port(reg_file);
    cdb.reserve_stations.bind(alu_RS);
    cdb.reserve_stations.bind(bu_RS);
    cdb.reserve_stations.bind(ls_RS);
    cdb.reserve_stations.bind(tensor_RS);

    ifetch IF("IF", cfg);
    idecode ID("ID", cfg);

    IF.clk(clk);
    ID.clk(clk);

    IF.jump_valid(jump_valid);
    ID.jump_valid(jump_valid);

    IF.jump_address(jump_address);
    ID.jump_address(jump_address);

    IF.branch_valid(branch_valid);
    ID.branch_valid(branch_valid);

    IF.branch_address(branch_address);

    stall.write(false);
    IF.stall_fetch(stall);
    ID.stall_decode(stall);

    IF.instructions.bind(IF_ID);
    ID.in_instructions.bind(IF_ID);

    ID.out_instructions.bind(ID_IQ);

    IF.set_insn(vctr_test_insns());

    // Monitor monitor("Monitor", issueDepth);
    // monitor.clk(clk);
    // monitor.insns.bind(ID_IQ);
    issue_queue iQueue("Issue-Queue", cfg);
    iQueue.clk(clk);
    iQueue.stall(stall);
    iQueue.instructions(ID_IQ);
    iQueue.future_file(reg_file);
    iQueue.alu_rs(alu_RS);
    iQueue.branch_known(branch_known);
    iQueue.branch_valid(branch_valid);
    iQueue.branch_rs(bu_RS);
    iQueue.lsu_rs(ls_RS);
    iQueue.tensor_rs(tensor_RS);

    alu _alu("ALU", cfg);
    _alu.clk(clk);
    _alu.reserve_station(alu_RS);
    _alu.data_bus(cdb);

    branch_unit bu("Branch-Unit", cfg);
    bu.clk(clk);
    bu.reserve_station(bu_RS);
    bu.branch_known(branch_known);
    bu.branch_valid(branch_valid);
    bu.branch_address(branch_address);

    scalar_memory sclr_mem("scalar-memory", cfg);

    load_store_unit lsu("load-store-unit", cfg);
    lsu.clk(clk);
    lsu.reserve_station(ls_RS);
    lsu.memory(sclr_mem);
    lsu.data_bus(cdb);

    address_generate_unit agu("address-generate-unit", cfg);
    agu.clk(clk);
    agu.reserve_station(tensor_RS);
    agu.matrix_mq(matrix_mq);
    agu.vector_mq(vector_mq);

    assert(cfg["scratchpad_design"].as<string>() == "unified" && ", only unified scrachpad is supported now");
    shared_ptr<RAM> buffer;
    {
        std::size_t size = 1 << cfg["scratchpad"]["log_size_per_channel"].as<std::size_t>();
        size *= cfg["scratchpad"]["nchannel"].as<std::size_t>();
        buffer.reset(new RAM(size));
    }
    ScratchpadHolder holder(cfg, buffer);

    vector_unit vctr_unit("vector-unit", cfg, holder);
    vctr_unit.clk(clk);
    vctr_unit.mem_queue_read(vector_mq);
    vctr_unit.mem_queue_commit(vector_mq);
    vctr_unit.retire_bus_port(r_bus);

    /* prepare data here */
    for (unsigned i = 0; i <= 4; ++i)
    {
        buffer->Memset(i + 1, i * 8, 8);
    }

    sc_start(300, SC_NS);

    /* check result here */
    for (unsigned i = 0; i <= 4; ++i)
    {
        Byte arr[8];
        buffer->CopyTo(arr, i * 8, 8);
        for (unsigned j = 0; j < 8; ++j)
            std::cout << static_cast<int>(arr[j]) << ' ';
        cout << endl;
    }

    // std::cout << "issued " << iQueue.GetIssueCount() << " instructions" << std::endl;

    return 0;
}