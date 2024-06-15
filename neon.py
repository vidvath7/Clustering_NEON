import torch
import numpy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the soft minpooling and soft maxpooling layers
def smin(X, s, dim=-1):
    return -(1/s)*torch.logsumexp(-s*X, dim=dim) + (1/s)*np.log(X.shape[dim])

def smax(X, s, dim=-1):
    return (1/s)*torch.logsumexp(s*X, dim=dim) - (1/s)*np.log(X.shape[dim])

class NeuralizedKMeans(torch.nn.Module):
    def __init__(self, kmeans):
        super().__init__()
        self.n_clusters = kmeans.n_clusters
        self.kmeans = kmeans
        K, D = kmeans.cluster_centers_.shape
        self.centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.double, device=device)
        self.W = torch.empty(K, K-1, D, dtype=torch.double, device=device)
        self.b = torch.empty(K, K-1, dtype=torch.double, device=device)
        for c in range(K):
            for kk in range(K-1):
                k = kk if kk < c else kk + 1
                self.W[c, kk] = 2 * (self.centroids[c] - self.centroids[k])
                self.b[c, kk] = (torch.norm(self.centroids[k])**2 - torch.norm(self.centroids[c])**2)

    def h(self, X):
        z = torch.einsum('ckd,nd->nck', self.W, X) + self.b
        return z

    def forward(self, X, c=None):
        h = self.h(X)
        out = h.min(-1).values
        if c is None:
            return out.max(-1).values
        else:
            return out[:, c]

def inc(z, eps=1e-9):
    return z + eps * (2 * (z >= 0) - 1)

def beta_heuristic(model, X):
    fc = model(X)
    return 1 / fc.mean()

def neon(model, X, beta):
    R = torch.zeros_like(X)
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, dtype=torch.double, device=device)
    for i in range(X.shape[0]):
        x = X[[i]]
        h = model.h(x)
        out = h.min(-1).values
        c = out.argmax()
        pk = torch.nn.functional.softmin(beta * h[:, c], dim=-1)
        Rk = out[:, c] * pk
        knc = [k for k in range(model.n_clusters) if k != c]
        Z = model.W[c] * (x - 0.5 * (model.centroids[[c]] + model.centroids[knc]))
        Z = Z / inc(Z.sum(-1, keepdims=True))
        R[i] = (Z * Rk.view(-1, 1)).sum(0)
    return R
