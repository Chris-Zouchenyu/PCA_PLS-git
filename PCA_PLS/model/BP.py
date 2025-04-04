from torch.nn import Linear,Sequential
import torch
class BP_model(torch.nn.Module):
    def __init__(self,k):
        super().__init__()
        self.model = Sequential(
            Linear(k,10),
            Linear(10,5),
            Linear(5,1)
        )
    def forward(self,x):
        x = self.model(x)
        return x

# 测试一下
# model = BP_model()
# x = torch.randn(401,10)
# y = model(x)
# print(y.shape)# (401,10) -> (401,1)

