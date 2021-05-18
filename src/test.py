import torch


if __name__ == '__main__':
    a = torch.randn((5,5))
    b = [0,1,2]
    c = [3,4]
    print(a)
    print(a[:, b])
    print(a[:, c])