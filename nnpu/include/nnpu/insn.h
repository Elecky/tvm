/*
the instruction set definitions of nnpu simulator
*/

#ifndef NNPU_INSN_H
#define NNPU_INSN_H

#include <nnpusim/insn.h>
#include <nnpusim/micro_code.h>

namespace nnpu
{

struct InsnDumper
{
    using result_type = void;
public:
    template<typename T>
    inline void operator()(const T& value, std::ostream& os)
    {
        value.Dump(os);
    }
};

}  // namespace nnpu

#endif