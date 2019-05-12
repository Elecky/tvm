<img src=https://raw.githubusercontent.com/tqchen/tvm.ai/master/images/logo/tvm-logo-small.png width=128/> Open Deep Learning Compiler Stack
==============================================

[![GitHub license](https://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![Build Status](http://ci.tvm.ai:8080/buildStatus/icon?job=tvm/master)](http://ci.tvm.ai:8080/job/tvm/job/master/)

[Documentation](https://docs.tvm.ai) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.ai/community.html) |
[Release Notes](NEWS.md)

TVM is a compiler stack for deep learning systems. It is designed to close the gap between the
productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends.
TVM works with deep learning frameworks to provide end to end compilation to different backends.
Checkout the [tvm stack homepage](https://tvm.ai/)  for more information.

License
-------
© Contributors Licensed under an [Apache-2.0](https://github.com/dmlc/tvm/blob/master/LICENSE) license.

Contribute to TVM
-----------------
TVM adopts apache committer model, we aim to create an open source project that is maintained and owned by the community.
Checkout the [Contributor Guide](https://docs.tvm.ai/contribute/)

Acknowledgement
---------------
We learnt a lot from the following projects when building TVM.
- [Halide](https://github.com/halide/Halide): TVM uses [HalideIR](https://github.com/dmlc/HalideIR) as data structure for
  arithmetic simplification and low level lowering. We also learnt and adapted some part of lowering pipeline from Halide.
- [Loopy](https://github.com/inducer/loopy): use of integer set analysis and its loop transformation primitives.
- [Theano](https://github.com/Theano/Theano): the design inspiration of symbolic scan operator for recurrence.

## 安装
基本安装步骤参考[TVM官方文档](https://docs.tvm.ai/install/from_source.html)
#### 本fork repo额外的依赖
1. 需要安装python yaml包。例如使用`pip install pyyaml`安装。
2. 需要编译并安装[yaml-cpp库](https://github.com/jbeder/yaml-cpp)。
3. 下载并按说明编译这个[LLVM repo](https://github.com/Elecky/llvm)。并在编译TVM时，在`build/config.cmake`中修改`set(USE_LLVM /path/to/your/llvm/bin/llvm-config)`，令它指向刚刚编译的LLVM。
4. 下载并编译安装[SystemC 2.3.3](https://accellera.org/images/downloads/standards/systemc/systemc-2.3.3.tar.gz)。建议在编译SystemC时使用C++14标准（`cmake -DCMAKE_CXX_STANDARD=14 ...`），否则可能出现加载libsystemc.so时符号找不到的错误。

## 运行NNPU demo
* `tvm/nnpu/tests`目录下有各个算子的测试代码，例如执行`python test_conv.2.py`执行卷积算子演示。使用`--sim S0/SC`参数选择功能级/行为级仿真器,例如`python test_conv.2.py --sim SC`。
* 配置文件`tvm/nnpu/config/nnpu_config.yaml`包含NNPU后端硬件属性的配置参数，也可以将其复制到`tvm/nnpu/`目录下。
