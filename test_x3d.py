import torch
import torch.nn as nn

model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
print("BLOCK 5:", model.blocks[5])
print("BLOCK 5 POOL:", model.blocks[5].pool)
print("BLOCK 5 PROJ:", model.blocks[5].proj)

# Test dummy input
dummy = torch.randn(1, 3, 40, 224, 224)
x = dummy
for i in range(5):
    x = model.blocks[i](x)
print("Out of block 4 shape:", x.shape)

x = model.blocks[5].pool(x)
print("After ProjectedPool shape:", x.shape)


