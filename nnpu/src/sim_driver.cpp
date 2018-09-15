/*
this is the base simulator driver of NNPU,
some common functionalities are implemented here.
also declares the base simulator interface.
*/

#include <nnpu/driver.h>

class NNPU_Simulator
{
public:
    const NNPUDevType devType;

    virtual ~NNPU_Simulator() = 0;
};