import nnvm
import nnvm.compiler
import nnvm.symbol as sym

data = sym.Variable('data')
weight = sym.Variable('weight')
c1 = sym.conv2d(data=data, weight=weight, use_bias=False,
                channels=32, kernel_size=(4,4))

p1 = sym.avg_pool2d(data=c1, pool_size=(4, 4))
s1 = sym.relu(data=p1)

compute_graph = nnvm.graph.create(s1)
print('-------original')
print(compute_graph.ir())
print('-------after')

deploy_graph, lib, params = nnvm.compiler.build(
    compute_graph, target='llvm', shape={'data': (1, 32, 32, 32)}, dtype='float32'
)
print(deploy_graph.ir())