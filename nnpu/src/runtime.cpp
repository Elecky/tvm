/*
NNPU runtime
*/

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <nnpu/driver.h>
#include <nnpu/insn.h>
#include <nnpu/runtime.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <vector>
#include <sstream>

#define DeclareAssembleFunc(funcName) void funcName(const vector<string>&, const vector<string>&, const string&)

static const bool kBufferCoherent = false;

namespace nnpu
{

/*!
 * ack: code from tvm.vta
 * \brief Data buffer represents data on CMA.
 */
struct DataBuffer
{
    /*! \return Virtual address of the data. */
    void *virt_addr() const
    {
        return data_;
    }
    /*! \return Physical address of the data. */
    uint32_t phy_addr() const
    {
        return phy_addr_;
    }
    /*!
    * \brief Invalidate the cache of given location in data buffer.
    * \param offset The offset to the data.
    * \param size The size of the data.
    */
    void InvalidateCache(size_t offset, size_t size)
    {
        if (!kBufferCoherent)
        {
            NNPUInvalidateCache(phy_addr_ + offset, size);
        }
    }
    /*!
    * \brief Invalidate the cache of certain location in data buffer.
    * \param offset The offset to the data.
    * \param size The size of the data.
    */
    void FlushCache(size_t offset, size_t size)
    {
        if (!kBufferCoherent)
        {
            NNPUFlushCache(phy_addr_ + offset, size);
        }
    }
    /*!
    * \brief Allocate a buffer of a given size.
    * \param size The size of the buffer.
    */
    static DataBuffer *Alloc(size_t size)
    {
        void *data = NNPUMemAlloc(size, 1);
        CHECK(data != nullptr);
        DataBuffer *buffer = new DataBuffer();
        buffer->data_ = data;
        buffer->phy_addr_ = NNPUMemGetPhyAddr(data);
        return buffer;
    }
    /*!
    * \brief Free the data buffer.
    * \param buffer The buffer to be freed.
    */
    static void Free(DataBuffer *buffer)
    {
        NNPUMemFree(buffer->data_);
        delete buffer;
    }
    /*!
    * \brief Create data buffer header from buffer ptr.
    * \param buffer The buffer pointer.
    * \return The corresponding data buffer header.
    */
    static DataBuffer *FromHandle(const void *buffer)
    {
        return const_cast<DataBuffer *>(
            reinterpret_cast<const DataBuffer *>(buffer));
    }

  private:
    /*! \brief The internal data. */
    void *data_;
    /*! \brief The physical address of the buffer, excluding header. */
    uint32_t phy_addr_;
};

using std::vector;
using std::string;

static string Trim(string str)
{
    auto first = str.find_first_not_of(" \t");
    if (first == string::npos)
    {
        return string();
    }
    else
    {
        string res = str.substr(first);
        auto last = res.find_last_not_of(" \t");
        res.resize(last + 1);
        return res;
    }
}

static vector<string> Split(const string &str, const std::unordered_set<char> &delim)
{
    vector<string> parts;
    string token;
    for (const auto &c : str)
    {
        if (delim.count(c) == 0)
        {
            token.push_back(c);
        }
        else
        {
            if (token.length() != 0)
            {
                parts.push_back(move(token));
                token = string();
            }
        }
    }

    if (token.size() != 0)
        parts.push_back(move(token));

    return parts;
}

class NNPUAssembler
{
public:
    NNPUAssembler();

    /*
     * \brief assemble the asm code, creating instructions.
     * \param asm_str the assembly code string.
     */
    void Assemble(string asm_str);

    /*
     * \brief check whether a line of assembly code is asm directive.
     * \param line: the line of assembly code.
     * \return true if this line is directive.
     */
    bool IsDirective(const string &line)
    {
        return line.length() > 0 &&
               line[0] == '.' &&
               !IsLabel(line);
    }

    /*
     * \brief check whether a line of assembly code is asm directive.
     * \param line: the line of assembly code.
     * \return true if this line is a label.
     */
    inline bool IsLabel(const string &line)
    {
        return line.length() > 0 && (*(line.end() - 1) == ':');
    }

    /*
     * \brief get the assembled insns. Note: this functions will do a copy.
     * \return vector of assembled insns.
     */
    inline vector<NNPUInsn> GetInsns() const
    {
        return insns;
    }

private:
    vector<NNPUInsn> insns;

    // label string to address
    std::unordered_map<string, std::size_t> labelAddr;

    // a relocation record struct.
    struct RelocRecord
    {
        std::size_t insnIdx;  // instruction index.
        std::size_t RelocPtr;  // relative address from the beginning of NNPUInsn struct 
                               // to offset field. 
        string label;  // the label it should be relocated to.
        bool IsRelative;  // relative or absolute relocate?
        uint32_t Base;  // base address if relative relocate required.
    };

    enum class Segment { text, data, bss };  // the enum which indicates segment.

    // relocation records.
    vector<RelocRecord> relocRecords;

    // subscribed handlers for different instructions, use unordered_map to do dispatching.
    std::unordered_map<string, 
                       void (NNPUAssembler::*)(
                                const vector<string>&,
                                const vector<string>&,
                                const string&)> instrHandler;
    
    /*
     * \brief parse a register operand, return the register No.
     * \param token: register operand
     * \return the register No.
     */
    static regNo_t parseReg(const string &token);

    /*
     * \brief parse a memory operand, checks whether its offset is immediate value or label.
     *        if offset is a label, corresponding relocation records will be created.
     * \param token: the operand token.
     * \param baseReg: return base regiser No. 
     * \param offset: return immediate offset.
     * \param rr: pointer to return relocation record,
     *            the base, label and isRelative will be set occordingly.
     *            caller should fill other fields and insert it into relocation list.
     * \return is label offset? that is, is relocation record needed.
     */
    bool parseMemOperand(const string &token, 
                         regNo_t &baseReg, uint32_t &offset,
                         RelocRecord *rr);

    inline static int32_t parseInt(const string &token)
    {
        return std::atoi(token.c_str());
    }

    inline static double parseDouble(const string &token)
    {
        return std::stod(token.c_str());
    }

    inline static bool parseBool(const string &token)
    {
        assert(token.length() == 1 && ", a boolean value should be either T or F");
        return token[0] == 'T';
    }

    // handlers for instructions.
    // signatures for all those functions are:
    // void (const vector<string> &functs, 
    //       const vector<string> &tokens, 
    //       const string &instr)
    // whether tokens are the line of instruction splited by space and comma,
    // and functs are the first token splitted by dot, instr are the original line of code.
    DeclareAssembleFunc(assembleLoad);
    DeclareAssembleFunc(assembleStore);
    DeclareAssembleFunc(assembleALUBinary);
    DeclareAssembleFunc(assembleALUUnary);
    DeclareAssembleFunc(assembleJump);
    DeclareAssembleFunc(assembleBZ);
    DeclareAssembleFunc(assembleDMA);
    DeclareAssembleFunc(assembleBufferLS);
    DeclareAssembleFunc(assembleVctrBinary);
    DeclareAssembleFunc(assembleRet);
    DeclareAssembleFunc(assembleMemset);
    DeclareAssembleFunc(assembleVDotV);
    DeclareAssembleFunc(assembleGEMM);
    DeclareAssembleFunc(assembleAccMemset);
    DeclareAssembleFunc(assembleCopyAccToBuffer);
    DeclareAssembleFunc(assembleMatImm);
    DeclareAssembleFunc(assembleVctrImm);
    DeclareAssembleFunc(assembleMatReduce);
    DeclareAssembleFunc(assembleVctrSclr);
    DeclareAssembleFunc(assembleVctrUnary);
    DeclareAssembleFunc(assembleVctrReduce);
    DeclareAssembleFunc(assembleMatRowDot);
    DeclareAssembleFunc(assembleMatBinary);
    DeclareAssembleFunc(assembleCopy);
    DeclareAssembleFunc(assembleMatVctr);

    static const std::unordered_set<char> tokenDelims;
    static const std::unordered_set<char> functDelims;
};

const std::unordered_set<char> NNPUAssembler::tokenDelims = {',', ' ', '\t'};
const std::unordered_set<char> NNPUAssembler::functDelims = {'.'};

NNPUAssembler::NNPUAssembler()
{
    instrHandler.insert({"Load", &NNPUAssembler::assembleLoad});
    instrHandler.insert({"Store", &NNPUAssembler::assembleStore});

    static const vector<string> aluBinaryOps 
        { "AddU", "SubU", "MulU", "DivU", "ModU", "SLTU",
          "SLT", "SEQ", "XOR", "And", "Or" };
    for (auto &item : aluBinaryOps)
    {
        instrHandler.insert({item, &NNPUAssembler::assembleALUBinary});
    }

    static const vector<string> aluUnaryOps
        { "AddIU", "MulIU", "DivIU", "ModIU", "SLTIU", "SLTI",
          "SEQI", "XORI", "AndI", "OrI", "SHLI" };
    for (auto &item : aluUnaryOps)
    {
        instrHandler.insert({item, &NNPUAssembler::assembleALUUnary});
    }

    instrHandler.insert({"Jump", &NNPUAssembler::assembleJump});
    instrHandler.insert({"BNEZ", &NNPUAssembler::assembleBZ});
    instrHandler.insert({"BEZ", &NNPUAssembler::assembleBZ});
    instrHandler.insert({"DMALoad", &NNPUAssembler::assembleDMA});
    instrHandler.insert({"DMAStore", &NNPUAssembler::assembleDMA});
    instrHandler.insert({"ScratchpadLoad", &NNPUAssembler::assembleBufferLS});
    instrHandler.insert({"ScratchpadStore", &NNPUAssembler::assembleBufferLS});

    static const vector<string> vctrBinaryOps
        { "VAddV", "VSubV", "VMulV", "VDivV", "VGTMV" };
    for (auto &item : vctrBinaryOps)
    {
        instrHandler.insert({item, &NNPUAssembler::assembleVctrBinary});
    }

    instrHandler.insert({"ret", &NNPUAssembler::assembleRet});

    instrHandler.insert({"Memset", &NNPUAssembler::assembleMemset});

    instrHandler.insert({"VDotV", &NNPUAssembler::assembleVDotV});

    instrHandler.insert({"GEMM", &NNPUAssembler::assembleGEMM});

    instrHandler.insert({"AccMemset", &NNPUAssembler::assembleAccMemset});

    instrHandler.insert({"CopyAccToBuffer", &NNPUAssembler::assembleCopyAccToBuffer});

    static const vector<string> matImmOps
        { "MAddI", "MMulI", "ISubM" };
    for (auto &item : matImmOps)
    {
        instrHandler.insert({item, &NNPUAssembler::assembleMatImm});
    }

    static const vector<string> vctrImmOps
        { "VAddI", "VSubI", "VMulI", "VDivI", "VGTMI", "ISubV",
          "IDivV" };
    for (auto &item : vctrImmOps)
    {
        instrHandler.insert({item, &NNPUAssembler::assembleVctrImm});
    }

    instrHandler.insert({"MReduceSumRow", &NNPUAssembler::assembleMatReduce});

    static const vector<string> vctrSclrOps
        { "VAddS", "VSubS", "VMulS", "VDivS", "VGTMS", "SSubV", "SDivV"};
    for (auto &item : vctrSclrOps)
    {
        instrHandler.insert({item, &NNPUAssembler::assembleVctrSclr});
    }

    static const vector<string> vctrUnaryOps
        { "VExp", "VLog" };
    for (auto &item : vctrUnaryOps)
    {
        instrHandler.insert({item, &NNPUAssembler::assembleVctrUnary});
    }

    static const vector<string> vctrReduceOps
        { "VReduceSum", "VReduceMax", "VReduceMin" };
    for (auto &item : vctrReduceOps)
    {
        instrHandler.insert({item, &NNPUAssembler::assembleVctrReduce});
    }

    instrHandler.insert({"MRowDot", &NNPUAssembler::assembleMatRowDot});

    static const vector<string> matBinaryOps
        {"MAddM", "MSubM", "MMulM"};
    for (auto &item : matBinaryOps)
    {
        instrHandler.insert({item, &NNPUAssembler::assembleMatBinary});
    }

    instrHandler.insert({"ScratchpadCopy", &NNPUAssembler::assembleCopy});

    static const vector<string> matVctrOps
        {"MAddV", "MSubV", "MMulV"};
    for (auto &item : matVctrOps)
    {
        instrHandler.insert({item, &NNPUAssembler::assembleMatVctr});
    }
}

// void NNPU_VReduceKey(uint32_t out1Addr, uint32_t out2Addr, uint32_t inAddr, uint32_t size, uint32_t mode)
// {
//     using Li = nnpu::LiInsn;
//     nnpu::InsnQueue* queue = nnpu::InsnQueue::ThreadLocal();

//     // load 3 addresses
//     queue->EmplaceBack(Li(0, out1Addr));
//     queue->EmplaceBack(Li(1, out2Addr));

//     queue->EmplaceBack(Li(2, inAddr));
//     // create a gemm instruction
//     nnpu::VReduceKeyInsn reducekey(0,1,2,size,ModeFromInt(mode));

//     queue->EmplaceBack(reducekey);
// }

// static bool DumpInsn = true;
// using tvm::runtime::TVMArgs;
// using tvm::runtime::TVMRetValue;
// static TVM_ATTRIBUTE_UNUSED auto &__register_dev__ =
//     ::tvm::runtime::Registry::Register("nnpu.set_dump", true)
//         .set_body([](TVMArgs args, TVMRetValue *rv) {
//             if (args.size() >= 1)
//                 DumpInsn = static_cast<bool>(args[0]);
//         });

// void NNPUSynchronize(uint32_t timeout)
void NNPUAssembler::Assemble(string asm_str)
{
    // clear all states.
    insns.clear();
    labelAddr.clear();
    relocRecords.clear();

    // start 'assembling'
    Segment segment = Segment::text;

    std::stringstream ss(asm_str);
    constexpr std::size_t bufferSize = 1024;
    std::unique_ptr<char[]> buffer(new char[bufferSize]);

    while (ss.getline(buffer.get(), bufferSize))
    {
        string raw = string(buffer.get());
        string line = Trim(raw);

        if (line.length() == 0)  // a empty line.
        {
            continue;
        }

        if (IsLabel(line))
        {
            CHECK(segment == Segment::text)
                << ", only text segment is supported now";
            // insert label into label address list,
            // remove tailing ':' from this line to get label.
            labelAddr.insert({line.substr(0, line.size() - 1), insns.size()});
        }
        else if (IsDirective(line))
        {
            CHECK(line != ".data" && line != ".bss")
                << ", data and bss segment is not supported now";
        }
        else
        {
            // this is an instruction.
            auto tokens = Split(line, tokenDelims);
            CHECK_GT(tokens.size(), 0);

            auto functs = Split(tokens[0], functDelims);
            CHECK_GT(functs.size(), 0);

            auto handleIt = instrHandler.find(functs[0]);
            CHECK(handleIt != instrHandler.end()) 
                << ", handler for instruction '" << functs[0]
                << "' is not found";
            
            auto handlePtr = handleIt->second;
            CHECK(handlePtr != nullptr)
                << ", nullptr function pointer met";
            (this->*handlePtr)(functs, tokens, line);
        }
    }
    // add a nop instruction in case of a label points to the end.
    insns.emplace_back(JumpInsn(0));

    // start relocating.
    for (const auto &rr : relocRecords)
    {
        auto it = labelAddr.find(rr.label);
        CHECK(it != labelAddr.end())
            << "relocating target label not found";
        uint32_t *ptr = reinterpret_cast<uint32_t*>(
                            (reinterpret_cast<void*>(&insns[rr.insnIdx]) + rr.RelocPtr));
        *(ptr) = rr.IsRelative ? it->second - rr.Base : it->second;
    }

    // std::cout << " assembled insn: ";
    // InsnDumper dumper;
    // for (auto &insn : insns)
    // {
    //     insn.Call(dumper, std::cout);
    //     std::cout << std::endl;
    // }

    labelAddr.clear();
    relocRecords.clear();
}

regNo_t NNPUAssembler::parseReg(const string &token)
{
    CHECK_GT(token.length(), 2)
        << ", a register token should be at least 2 characters"
        << ", given token = " << token;
    CHECK_EQ(token[0], '$') << ", register token always starts with a '$'"
        << ", given token = " << token;

    if (isdigit(token[1]))
    {
        std::stringstream ss;
        ss << token.c_str() + 1;
        regNo_t regNo;
        ss >> regNo;
        return regNo;
    }
    else if (token[1] == 'g' || token[1] == 'G')
    {
        std::stringstream ss;
        ss << token.c_str() + 2;
        regNo_t regNo;
        ss >> regNo;
        return regNo;
    }
    else 
    {
        // registers that has special names.
        static const std::unordered_map<string, regNo_t> nameToNo 
                    { {"zero", 0}, {"sp", 1}, {"fp", 2} };
        
        string nameLCase(token.begin() + 1, token.end());
        for (auto &c : nameLCase)
        {
            c = tolower(c);
        }

        auto it = nameToNo.find(nameLCase);
        CHECK(it != nameToNo.end())
            << ", unknown register name: " << nameLCase;
        return it->second;
    }
}

bool NNPUAssembler::parseMemOperand(const string &token, 
                                    regNo_t &baseReg, uint32_t &offset,
                                    RelocRecord *rr)
{
    auto lp = token.find('(');
    CHECK(lp != string::npos)  
        << ", '(' not found in memory address operand"
        << ", the desired syntax is offset($base)";

    baseReg = parseReg(token.substr(lp + 1, token.length() - lp - 2));
    
    if (lp != 0)
    {
        string offStr(token.substr(0, lp));
        std::stringstream ss(offStr);
        if (isdigit(token[0]))  // if offset is immediate value.
        {
            ss >> offset;
            return false;
        }
        else  // this is a label
        {
            CHECK(rr != nullptr) << ", relocation needed but rr is nullptr";
            rr->IsRelative = false;  // absolute relocate.
            rr->label = offStr;
            return true;
        }
    }
    else
        offset = 0;
        return false;
}

void NNPUAssembler::assembleLoad(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 3) << ", ilegal syntax: " << instr;;

    regNo_t rd = parseReg(tokens[1]);
    insns.emplace_back(SclrLoadInsn(rd, 0, 0));

    SclrLoadInsn &insn = insns.back().SclrLoad;
    parseMemOperand(tokens[2], insn.AddrReg, insn.Offset, nullptr);
}

void NNPUAssembler::assembleStore(
        const vector<string> &functs, 
        const vector<string> &tokens, 
        const string &instr)
{
    CHECK_EQ(tokens.size(), 3) << ", ilegal syntax: " << instr;;

    regNo_t rs = parseReg(tokens[1]);
    insns.emplace_back(SclrStoreInsn(rs, 0, 0));

    SclrStoreInsn &insn = insns.back().SclrStore;
    parseMemOperand(tokens[2], insn.AddrReg, insn.Offset, nullptr);
}

void NNPUAssembler::assembleALUBinary(
        const vector<string> &functs, 
        const vector<string> &tokens, 
        const string &instr)
{
    static const std::unordered_map<string, ALUBinaryOp> Ops 
        {   {"AddU", ALUBinaryOp::Add}, {"SLT", ALUBinaryOp::SLT},
            {"SubU", ALUBinaryOp::Sub}, {"MulU", ALUBinaryOp::Mul},
            {"DivU", ALUBinaryOp::DivU}, {"ModU", ALUBinaryOp::ModU},
            {"SLTU", ALUBinaryOp::SLTU}, {"SEQ", ALUBinaryOp::SEQ},
            {"XOR", ALUBinaryOp::XOR}, {"And", ALUBinaryOp::And},
            {"Or", ALUBinaryOp::Or}
        };

    CHECK_EQ(tokens.size(), 4) << ", ilegal syntax: " << instr;;
    auto it = Ops.find(tokens[0]);
    CHECK(it != Ops.end()) << ", instruction operation mapping not found";

    insns.emplace_back(
            ALUBinaryInsn(parseReg(tokens[1]), parseReg(tokens[2]),
                          parseReg(tokens[3]), it->second));
}

void NNPUAssembler::assembleALUUnary(
        const vector<string> &functs, 
        const vector<string> &tokens, 
        const string &instr)
{
    static const std::unordered_map<string, ALURegImmOp> Ops
        {   {"AddIU", ALURegImmOp::AddIU}, {"MulIU", ALURegImmOp::MulIU},
            {"DivIU", ALURegImmOp::DivIU}, {"ModIU", ALURegImmOp::ModIU},
            {"SLTIU", ALURegImmOp::SLTIU}, {"SLTI", ALURegImmOp::SLTI},
            {"SEQI", ALURegImmOp::SEQI},   {"XORI", ALURegImmOp::XORI},
            {"AndI", ALURegImmOp::AndI},   {"OrI", ALURegImmOp::OrI},
            {"SHLI", ALURegImmOp::SHLI}
        };
    
    CHECK_EQ(tokens.size(), 4) << ", ilegal syntax: " << instr;
    auto it = Ops.find(tokens[0]);
    CHECK(it != Ops.end()) << ", instruction operation mapping not found";

    insns.emplace_back(
            ALURegImmInsn(parseReg(tokens[1]), parseReg(tokens[2]),
                          std::atoi(tokens[3].c_str()), it->second));
}

void NNPUAssembler::assembleJump(
        const vector<string> &functs, 
        const vector<string> &tokens, 
        const string &instr)
{
    CHECK_EQ(tokens.size(), 2) << ", ilegal syntax: " << instr;
    insns.emplace_back(JumpInsn(0));

    RelocRecord rr;
    rr.Base = insns.size() - 1;
    rr.IsRelative = true;
    rr.label = tokens[1];
    rr.RelocPtr = offsetof(NNPUInsn, Jump.Offset);
    rr.insnIdx = insns.size() - 1;
    relocRecords.push_back(rr);
}

void NNPUAssembler::assembleBZ(
        const vector<string> &functs, 
        const vector<string> &tokens, 
        const string &instr)
{
    CHECK_EQ(tokens.size(), 3) << ", ilegal syntax: " << instr;

    RelocRecord rr;
    rr.Base = insns.size();
    rr.IsRelative = true;
    rr.label = tokens[2];
    rr.insnIdx = insns.size();

    if (tokens[0] == "BNEZ")
    {
        insns.emplace_back(BNEZInsn(0, parseReg(tokens[1])));
        rr.RelocPtr = offsetof(NNPUInsn, BNEZ.Offset);
    }
    else if (tokens[0] == "BEZ")
    {
        insns.emplace_back(BEZInsn(0, parseReg(tokens[1])));
        rr.RelocPtr = offsetof(NNPUInsn, BEZ.Offset);
    }
    else
        LOG(ERROR) << "unhandled branch type";
    
    relocRecords.push_back(rr);
}

void NNPUAssembler::assembleDMA(
        const vector<string> &functs, 
        const vector<string> &tokens, 
        const string &instr)
{
    CHECK_EQ(tokens.size(), 5) << ", ilegal syntax: " << instr;
    DMADIR dir;
    if (tokens[0] == "DMALoad")
        dir = DMADIR::HtoD;
    else
        dir = DMADIR::DtoH;
    
    insns.emplace_back(
            DMACopyInsn(dir, parseReg(tokens[1]), parseReg(tokens[2]),
                        parseReg(tokens[3]), parseReg(tokens[4])));
}

void NNPUAssembler::assembleBufferLS(
        const vector<string> &functs, 
        const vector<string> &tokens, 
        const string &instr)
{
    CHECK_EQ(tokens.size(), 4) << ", ilegal syntax: " << instr;

    LSDIR dir = tokens[0] == "ScratchpadLoad" ? LSDIR::Load : LSDIR::Store;
    insns.emplace_back(
            BufferLSInsn(dir, parseReg(tokens[1]), parseReg(tokens[2]),
                         parseReg(tokens[3])));
}

void NNPUAssembler::assembleVctrBinary(
        const vector<string> &functs, 
        const vector<string> &tokens, 
        const string &instr)
{
    CHECK_EQ(tokens.size(), 4) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 3) << ", ilegal syntax: " << instr;

    static const std::unordered_map<string, VctrBinaryOp> Ops
        { {"VAddV", VctrBinaryOp::Add}, {"VSubV", VctrBinaryOp::Sub},
          {"VMulV", VctrBinaryOp::Mul}, {"VDivV", VctrBinaryOp::Div},
          {"VGTMV", VctrBinaryOp::GTM} };
    auto it = Ops.find(functs[0]);
    CHECK(it != Ops.end()) << ", vector binary op not found for " << functs[0];

    insns.emplace_back(
            VctrBinaryInsn(it->second, parseReg(tokens[1]),
                           parseReg(tokens[2]), parseReg(tokens[3]),
                           parseInt(functs[1]), 
                           ModeFromInt(parseInt(functs[2]) ))
    );
}

void NNPUAssembler::assembleRet(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    insns.emplace_back(StallInsn());
}

void NNPUAssembler::assembleMemset(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 5) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 2) << ", ilegal syntax: " << instr;

    insns.emplace_back(
        MemsetInsn(parseReg(tokens[1]) /* addr */, parseReg(tokens[2]) /* nUnit */,
                    parseReg(tokens[3]) /* stride */, ModeFromInt(parseInt(functs[1])),
                    parseDouble(tokens[4]) /* value to set */)
    );
}

void NNPUAssembler::assembleVDotV(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 4) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 3) << ", ilegal syntax: " << instr;

    insns.emplace_back(
        VctrDotProdInsn(parseReg(tokens[1]) /*out addr*/, parseReg(tokens[2]) /*in1 addr*/,
                        parseReg(tokens[3]) /*in2 addr*/, parseInt(functs[1]) /*size*/,
                        ModeFromInt(parseInt(functs[2])) /*mode*/)
    );
}

void NNPUAssembler::assembleGEMM(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 7) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 7) << ", ilegal syntax: " << instr;

    insns.emplace_back(
        GemmInsn(parseInt(functs[1]) /*nRowOut*/, parseInt(functs[2]) /*factor*/,
                parseInt(functs[3]) /*nColOut*/, 
                parseReg(tokens[1]) /*out addr*/, parseReg(tokens[2]),
                parseReg(tokens[3]) /*in1 addr*/, parseReg(tokens[4]),
                parseReg(tokens[5]) /*in2 addr*/, parseReg(tokens[6]),
                ModeFromInt(parseInt(functs[4])),
                parseBool(functs[5]) /*toAcc*/, parseBool(functs[6]) /*doAcc*/)
    );
}

void NNPUAssembler::assembleAccMemset(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 4) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 4) << ", ilegal syntax: " << instr;

    insns.emplace_back(
        AccMemsetInsn(parseInt(functs[1]) /*nRow*/, parseInt(functs[2]) /*nCol*/,
                    parseReg(tokens[1]) /*addr*/, parseReg(tokens[2]) /*stride*/,
                    ModeFromInt(parseInt(functs[3])) /*mode*/,
                    parseDouble(tokens[3]) /*value*/)
    );
}

void NNPUAssembler::assembleCopyAccToBuffer(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 6) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 2) << ", ilegal syntax: " << instr;

    insns.emplace_back(
        CopyAcc2BufInsn(parseReg(tokens[1]) /*dst addr*/, parseReg(tokens[2]),
                        parseReg(tokens[3]) /*src addr*/, parseReg(tokens[4]),
                        parseReg(tokens[5]), parseInt(functs[1]))
    );
}

void NNPUAssembler::assembleMatImm(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 4) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 4) << ", ilegal syntax: " << instr;

    static const std::unordered_map<string, MatImmOp> Ops
    { {"MAddI", MatImmOp::Add}, {"MMulI", MatImmOp::Mul},
      {"ISubM", MatImmOp::RSub} };
    
    auto it = Ops.find(functs[0]);
    CHECK(it != Ops.end()) << ", unhandled MatImm op: " << functs[0];

    insns.emplace_back(
        MatImmInsn(it->second, parseReg(tokens[1]) /*out addr*/,
                    parseReg(tokens[2]) /*in addr*/, parseDouble(tokens[3]) /*imm value*/,
                    parseInt(functs[1]) /*nRow*/, parseInt(functs[2]) /*nCol*/,
                    ModeFromInt(parseInt(functs[3])) /*mode*/)
    );
}

void NNPUAssembler::assembleVctrImm(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 4) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 3) << ", ilegal syntax: " << instr;

    static const std::unordered_map<string, VctrImmOp> Ops
    { 
        {"VAddI", VctrImmOp::Add}, {"VSubI", VctrImmOp::Sub},
        {"VMulI", VctrImmOp::Mul}, {"VDivI", VctrImmOp::Div},
        {"VGTMI", VctrImmOp::GTM}, {"ISubV", VctrImmOp::RSub},
        {"IDivV", VctrImmOp::RDiv}
    };
    
    auto it = Ops.find(functs[0]);
    CHECK(it != Ops.end()) << ", unhandled VctrImm op: " << functs[0];

    insns.emplace_back(
        VctrImmInsn(it->second /*op*/, parseReg(tokens[1]) /*out addr*/,
                    parseReg(tokens[2]) /*in addr*/, parseDouble(tokens[3]),
                    parseInt(functs[1]) /*size*/, 
                    ModeFromInt(parseInt(functs[2])) /*mode*/)
    );
}

void NNPUAssembler::assembleMatReduce(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 4) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 6) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs[0], "MReduceSumRow") << ", unhandled matrix reduce op: " << functs[0];

    insns.emplace_back(
        MatReduceRowInsn(parseReg(tokens[1]) /*out addr*/, parseReg(tokens[2]) /*in addr*/,
                        parseReg(tokens[3]) /*in stride*/,
                        ReduceOp::Sum,
                        parseInt(functs[1]) /*nRow*/, parseInt(functs[2]) /*nCol*/,
                        parseBool(functs[4]) /*toAcc*/, parseBool(functs[5]) /*doAcc*/,
                        ModeFromInt(parseInt(functs[3])) /*mode*/)
    );
}

void NNPUAssembler::assembleVctrSclr(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 4) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 3) << ", ilegal syntax: " << instr;

    static const std::unordered_map<string, VctrSclrOp> Ops
        {   {"VAddS", VctrSclrOp::Add}, {"VSubS", VctrSclrOp::Sub},
            {"VMulS", VctrSclrOp::Mul}, {"VDivS", VctrSclrOp::Div},
            {"VGTMS", VctrSclrOp::GTM}, {"SSubV", VctrSclrOp::RSub},
            {"SDivV", VctrSclrOp::RDiv},
        };
    auto it = Ops.find(functs[0]);
    CHECK(it != Ops.end()) << ", unhandled vector scalar op: " << functs[0];

    insns.emplace_back(
        VctrSclrInsn(parseReg(tokens[1]) /*out addr*/, parseReg(tokens[2]) /*vctr in addr*/,
                    parseReg(tokens[3]) /*sclr in addr*/,
                    parseInt(functs[1]) /*size*/, it->second /*op*/,
                    ModeFromInt(parseInt(functs[2])) /*mode*/)
    );
}

void NNPUAssembler::assembleVctrUnary(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 3) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 3) << ", ilegal syntax: " << instr;

    static const std::unordered_map<string, VctrUnaryOp> Ops
        { {"VExp", VctrUnaryOp::Exp}, {"VLog", VctrUnaryOp::Log} };
    auto it = Ops.find(functs[0]);
    CHECK(it != Ops.end()) << ", unhandled vector unary op: " << functs[0];

    insns.emplace_back(
        VctrUnaryInsn(it->second /*op*/, 
                        parseReg(tokens[1]) /*out addr*/, parseReg(tokens[2]) /*in addr*/,
                        parseInt(functs[1]) /*size*/,
                        ModeFromInt(parseInt(functs[2])) /*mode*/)
    );
}

void NNPUAssembler::assembleVctrReduce(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 3) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 3) << ", ilegal syntax: " << instr;

    static const std::unordered_map<string, ReduceOp> Ops
        { 
            {"VReduceSum", ReduceOp::Sum}, {"VReduceMax", ReduceOp::Max}, 
            {"VReduceMin", ReduceOp::Min}
        };
    auto it = Ops.find(functs[0]);
    CHECK(it != Ops.end()) << ", unhandled vector reduce op: " << functs[0];

    insns.emplace_back(
        VctrReduceInsn(parseReg(tokens[1]) /*out addr*/, parseReg(tokens[2]) /*in addr*/,
                        it->second, parseInt(functs[1]) /*size*/,
                        ModeFromInt(parseInt(functs[2]))) /*mode*/
    );
}

void NNPUAssembler::assembleMatRowDot(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 6) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 6) << ", ilegal syntax: " << instr;

    insns.emplace_back(
        MatRowDotInsn(parseReg(tokens[1]) /*out addr*/, 
                        parseReg(tokens[2]) /*in1 addr*/, parseReg(tokens[3]),
                        parseReg(tokens[4]) /*in2 addr*/, parseReg(tokens[5]),
                        parseInt(functs[1]) /*nRow*/, parseInt(functs[2]) /*nCol*/,
                        parseBool(functs[4]) /*toAcc*/, parseBool(functs[5]) /*doAcc*/,
                        ModeFromInt(parseInt(functs[3])) /*mode*/)
    );
}

void NNPUAssembler::assembleMatBinary(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 7) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 4) << ", ilegal syntax: " << instr;

    static const std::unordered_map<string, MatBinaryOp> Ops
        {
            {"MAddM", MatBinaryOp::Add}, {"MSubM", MatBinaryOp::Sub},
            {"MMulM", MatBinaryOp::Mul}
        };
    auto it = Ops.find(functs[0]);
    CHECK(it != Ops.end()) << ", unhandled matrix binary op: " << functs[0];

    insns.emplace_back(
        MatBinaryInsn(parseReg(tokens[1]) /*out addr*/, parseReg(tokens[3]) /*in1 addr*/,
                    parseReg(tokens[5]) /*in2 addr*/,
                    parseReg(tokens[2]) /*out stride*/, parseReg(tokens[4]) /*in1 stride*/,
                    parseReg(tokens[6]) /*in2 stride*/,
                    it->second /*op*/,
                    parseInt(functs[1]) /*nRow*/, parseInt(functs[2]) /*nCol*/,
                    ModeFromInt(parseInt(functs[3])) /*mode*/)
    );
}

void NNPUAssembler::assembleCopy(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 6) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 2) << ", ilegal syntax: " << instr;

    insns.emplace_back(
        BufferCopyInsn(parseReg(tokens[1]) /*dst addr*/, parseReg(tokens[2]) /*dst stride*/,
                        parseReg(tokens[3]) /*src addr*/, parseReg(tokens[4]) /*src stride*/,
                        parseReg(tokens[5]) /*nUnits*/,
                        parseInt(functs[1]) /*per unit bytes*/)
    );
}

void NNPUAssembler::assembleMatVctr(const vector<string> &functs, 
                                 const vector<string> &tokens,
                                 const string &instr)
{
    CHECK_EQ(tokens.size(), 6) << ", ilegal syntax: " << instr;
    CHECK_EQ(functs.size(), 4) << ", ilegal syntax: " << instr;

    static const std::unordered_map<string, MatVctrOp> Ops
        {
            {"MAddV", MatVctrOp::Add }, {"MSubV", MatVctrOp::Sub },
            {"MMulV", MatVctrOp::Mul }
        };
    auto it = Ops.find(functs[0]);
    CHECK(it != Ops.end()) << ", unhandled matrix-vector op: " << functs[0];

    insns.emplace_back(
        MatVctrInsn(parseReg(tokens[1]) /*mat-out addr*/, parseReg(tokens[2]) /*stride*/,
                    parseReg(tokens[3]) /*mat-in addr*/, parseReg(tokens[4]) /*stride*/,
                    parseReg(tokens[5]) /*vctr-in addr*/,
                    it->second /*op*/,
                    parseInt(functs[1]) /*nRow*/, parseInt(functs[2]) /*nCol*/,
                    ModeFromInt(parseInt(functs[3])) /*mode*/)
    );
}

}  // end namespace nnpu

// the following 3 functions are from tvm.vta, used for managing driver buffers.
void *NNPUBufferAlloc(size_t size)
{
    return nnpu::DataBuffer::Alloc(size);
}

void NNPUBufferFree(void *buffer)
{
    return nnpu::DataBuffer::Free(nnpu::DataBuffer::FromHandle(buffer));
}

void NNPUBufferCopy(const void *from,
                    size_t from_offset,
                    void *to,
                    size_t to_offset,
                    size_t size,
                    int kind_mask)
{
    nnpu::DataBuffer *from_buffer = nullptr;
    nnpu::DataBuffer *to_buffer = nullptr;

    if (kind_mask & 2)  // source is accelerator
    {
        from_buffer = nnpu::DataBuffer::FromHandle(from);
        from = from_buffer->virt_addr();
    }
    if (kind_mask & 1)  // destination is accelerator
    {
        to_buffer = nnpu::DataBuffer::FromHandle(to);
        to = to_buffer->virt_addr();
    }
    if (from_buffer)
    {
        from_buffer->InvalidateCache(from_offset, size);
    }

    memcpy(static_cast<char *>(to) + to_offset,
           static_cast<const char *>(from) + from_offset,
           size);
    if (to_buffer)
    {
        to_buffer->FlushCache(to_offset, size);
    }
}

using std::vector;
using std::string;

static bool DumpInsn = false;
using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;

static TVM_ATTRIBUTE_UNUSED auto &__register_set_dump_ =
    ::tvm::runtime::Registry::Register("nnpu.set_dump", true)
        .set_body([](TVMArgs args, TVMRetValue *rv) {
            if (args.size() >= 1)
                DumpInsn = static_cast<bool>(args[0]);
        });

static uint32_t NNPU_Handle2PhyAddr(void *handle)
{
    auto buffer = nnpu::DataBuffer::FromHandle(handle);
    return buffer->phy_addr();
}

static TVM_ATTRIBUTE_UNUSED auto &__register_handleTophyAddr_ =
    ::tvm::runtime::Registry::Register("nnpu.handleToPhyAddr", true)
        .set_body([](TVMArgs args, TVMRetValue *rv) {
            CHECK(args.num_args >= 1) << ", ecpected one argument";
            CHECK(rv != nullptr) << ", empty return address";

            (*rv) = static_cast<int64_t>(
                        NNPU_Handle2PhyAddr(static_cast<void*>(args[0])));
        });

extern "C" void NNPU_AssembleAndRun(
                    string asm_code, 
                    string func_name, 
                    int coproc_scope,
                    std::vector<int32_t> args)
{
    // auto &os = LOG(INFO);
    // os << "NNPU runtime function: NNPU_AssembleAndRun";
    // os << "\n call args:\n  [";
    // for (auto it : args)
    // {
    //     os << it << ", ";
    // }
    // os << "]\n coproc scope = " << coproc_scope;
    // os << "\n calling function [" << func_name;
    // os << "] in asm code: \n";
    // os << asm_code;

    // os << "begin assembling\n";

    nnpu::NNPUAssembler assembler;
    assembler.Assemble(asm_code);

    // assign arguments.
    auto sim = nnpu::Simulator::ThreadLocal();
    uint32_t fp = sim->GetSclrMemSize() - args.size() * sizeof(uint32_t);
    // LOG(INFO) << "FP = " << fp << std::endl;
    for (std::size_t i = 0; i != args.size(); ++i)
    {
        sim->WriteSclrMem(fp + i * sizeof(uint32_t), args[i]);
    }
    sim->WriteRegister(0, 0);
    sim->WriteRegister(1, fp);
    sim->WriteRegister(2, fp);

    auto insns = assembler.GetInsns();

    if (DumpInsn)
    {
        nnpu::InsnDumper dumper;
        auto &os = LOG(INFO) << "Dumping instructions: ";
        for (auto &insn : insns)
        {
            insn.Call(dumper, os);
            os << '\n';
        }
    }

    sim->Run(insns);
}

static TVM_ATTRIBUTE_UNUSED auto &__register_run_ =
    ::tvm::runtime::Registry::Register("nnpu.assemble_and_run", true)
        .set_body([](TVMArgs args, TVMRetValue *rv) {
            CHECK_GE(args.num_args, 3) << ", ecpected at least 3 arguments";
            CHECK_EQ(args.type_codes[0], kStr) 
                << ", expecting 1st argument to be assembly code [string]";
            CHECK_EQ(args.type_codes[1], kStr)
                << ", expecting 2nd argument to be function name [string]";
            CHECK_EQ(args.type_codes[2], kDLInt)
                << ", expecting 3rd argument to be coproc scope [int]";

            std::vector<int32_t> dev_args;  // arguments to be passed to device function.
            for (int i = 3; i < args.num_args; ++i)
            {
                CHECK_EQ(args.type_codes[i], kDLInt)
                    << ", only int type arguments can be passed to NNPU device";
                dev_args.push_back(static_cast<int32_t>(args[i]));
            }

            NNPU_AssembleAndRun(args[0].operator std::__cxx11::string(), 
                                args[1].operator std::__cxx11::string(),
                                static_cast<int>(args[2]),
                                dev_args);
        });