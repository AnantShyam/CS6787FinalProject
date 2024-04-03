import numpy as np
import torch 
import helpers

class Newton_Accelerated:

    def __init__(self, hessian, m, n):
        self.hessian = hessian # untransformed hessian 
        self.m = m 
        self.n = n

    def reshape_hessian(self):
        # hessian is of dimension mn x mn
        hessian_shape = self.hessian.shape
        assert self.m * self.n == hessian_shape[0]
        assert len(hessian_shape) == 2 and hessian_shape[0] == hessian_shape[1] # assert that hessian is a square matrix
        
        four_dim_hessian = torch.reshape(self.hessian, (self.m, self.n, self.m, self.n))
        non_symmetric_hessian = torch.reshape(four_dim_hessian, (self.m ** 2, self.n ** 2))
        return non_symmetric_hessian


if __name__ == "__main__":
    pass

