import torch
from model import X3DFreeKickModel

model = X3DFreeKickModel()
dummy_input = torch.randn(1, 3, 40, 224, 224)
x = dummy_input
for i in range(5):
    x = model.model.blocks[i](x)
print("Out of block 4:", x.shape)
x = model.model.blocks[5].pool(x)
print("After pool:", x.shape)
