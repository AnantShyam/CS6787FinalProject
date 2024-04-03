import numpy as np
import torch 
import helpers

class Newton_Accelerated:

    def __init__(self, hessian):
        self.hessian = hessian # untransformed hessian 

    def reshape_hessian(self, hessian, m, n):
        # hessian is of dimension mn x mn
        hessian_shape = hessian.shape
        assert m * n == hessian_shape[0]
        assert len(hessian_shape) == 2 and hessian_shape[0] == hessian_shape[1] # assert that hessian is a square matrix
        
        four_dim_hessian = torch.reshape(hessian, (m, n, m, n))
        non_symmetric_hessian = torch.reshape(four_dim_hessian, (m ** 2, n ** 2))
        return non_symmetric_hessian


if __name__ == "__main__":
    pass

