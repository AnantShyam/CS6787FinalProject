import torch 
import helpers
import a9a_analysis

class Newton_Accelerated:

    def __init__(self, hessian, m, n):
        self.hessian = hessian # untransformed hessian 
        self.m = m 
        self.n = n

    def approximate_hessian(self):
        # hessian is of dimension mn x mn

        # do reshaping of hessian here
        hessian_shape = self.hessian.shape
        assert self.m * self.n == hessian_shape[0]
        assert len(hessian_shape) == 2 and hessian_shape[0] == hessian_shape[1] # assert that hessian is a square matrix
        
        four_dim_hessian = torch.reshape(self.hessian, (self.m, self.n, self.m, self.n))
        non_symmetric_hessian = torch.reshape(four_dim_hessian, (self.m ** 2, self.n ** 2))
        U, S, V = torch.svd(non_symmetric_hessian)
        A, B = U[:, 0], V[:, 0]
        A, B = torch.reshape(A, (A.shape[0], 1)), torch.reshape(B, (B.shape[0], 1))

        sigma_max_sqrt = torch.sqrt(torch.max(S))

        A = sigma_max_sqrt * A 
        B = sigma_max_sqrt * B 
        approximate_hessian = A @ B.T 

        approximate_hessian = torch.reshape(approximate_hessian, hessian_shape)

        assert approximate_hessian.shape == hessian_shape
        return approximate_hessian



if __name__ == "__main__":
    
    # generate random matrix for testing purposes
    hessian = torch.rand(6, 6)
    newton_acc = Newton_Accelerated(hessian, 2, 3)
    approx_hessian = newton_acc.approximate_hessian()

