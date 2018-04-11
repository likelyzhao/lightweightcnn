import mxnet as mx
from importlib import import_module
net = import_module('symbols.'+'mobilenet')
sym = net.get_symbol(3)
sym.save('symbol.symbol')
