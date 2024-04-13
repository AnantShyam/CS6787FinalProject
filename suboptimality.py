import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, vstack

def logistic_loss(w, X, y, mu):
    z = X.dot(w)
    sigmoid = 1 / (1 + np.exp(-z))
    loss = -np.mean(y * np.log(sigmoid + 1e-10) + (1 - y) * np.log(1 - sigmoid + 1e-10)) + 0.5 * mu * np.linalg.norm(w)**2
    grad = X.T.dot(sigmoid - y) / X.shape[0] + mu * w
    return loss, grad

def gradient_descent_step(w, X, y, mu, learning_rate=0.01):
    loss, grad = logistic_loss(w, X, y, mu)
    w_new = w - learning_rate * grad
    return w_new, loss

def load_data(path, num_features=123):
    with open(path, 'r') as file:
        lines = file.readlines()

    data = vstack([csr_matrix(([1], ([0], [int(f.split(':')[0]) - 1])), shape=(1, num_features)) for line in lines if line.strip() for f in line.split()[1:]])
    labels = np.array([1 if line.split()[0] == '+1' else 0 for line in lines if line.strip()]).reshape(-1, 1)

    return data, labels

X, y = load_data("/Users/cba/Desktop/cs_6787/final_project/CS6787FinalProject/data/a9a_train.txt")
n_samples, n_features = X.shape
w = np.zeros(n_features) 

mu = 1 / n_samples 
losses = []

for it in range(100):
    w, loss = gradient_descent_step(w, X, y, mu)
    losses.append(loss)
    mu *= 0.95  # Gradually decrease regularization


plt.figure(figsize=(10, 6))
plt.plot(losses, label='Suboptimality')
plt.xlabel('Iterations')
plt.ylabel('Empirical Risk')
plt.title('Suboptimality on Empirical Risk vs Time')
plt.legend()
plt.grid(True)
plt.show()
