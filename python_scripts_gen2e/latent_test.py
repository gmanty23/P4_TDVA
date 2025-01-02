import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npmat
import warnings

from sklearn.datasets import load_digits
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
import tqdm

# Euclidean distance computation
def pairwise_distances(X):
    return np.sum((X[None, :] - X[:, None])**2, 2)

# Locating the points in a Gaussian depending on their distances
def p_conditional(dists, sigmas):
    e = np.exp(-dists / (2 * np.square(sigmas.reshape((-1,1)))))
    np.fill_diagonal(e, 0.)
    e += 1e-8
    return e / e.sum(axis=1).reshape([-1,1])

def perp(condi_matr):
    ent = -np.sum(condi_matr * np.log2(condi_matr), 1)
    return 2 ** ent

def find_sigmas(dists, perplexity):
    found_sigmas = np.zeros(dists.shape[0])
    for i in range(dists.shape[0]):
        func = lambda sig: perp(p_conditional(dists[i:i+1, :], np.array([sig])))
        found_sigmas[i] = search(func, perplexity)
    return found_sigmas

def search(func, goal, tol=1e-10, max_iters=1000, lowb=1e-20, uppb=10000):
    for _ in range(max_iters):
        guess = (uppb + lowb) / 2.
        val = func(guess)

        if val > goal:
            uppb = guess
        else:
            lowb = guess

        if np.abs(val - goal) <= tol:
            return guess

    warnings.warn(f"\nSearch couldn't find goal, returning {guess} with value {val}")
    return guess

def q_joint(y):
    dists = pairwise_distances(y)
    nom = 1 / (1 + dists)
    np.fill_diagonal(nom, 0.)
    return nom / np.sum(np.sum(nom))

def gradient(P, Q, y):
    (n, no_dims) = y.shape
    pq_diff = P - Q
    y_diff = np.expand_dims(y,1) - np.expand_dims(y,0) 

    dists = pairwise_distances(y)
    aux = 1 / (1 + dists)
    return 4 * (np.expand_dims(pq_diff, 2) * y_diff * np.expand_dims(aux,2)).sum(1)

def m(t):
    return 0.5 if t < 250 else 0.8

def p_joint(X, perp):
    N = X.shape[0]
    dists = pairwise_distances(X)
    sigmas = find_sigmas(dists, perp)
    p_cond = p_conditional(dists, sigmas)
    return (p_cond + p_cond.T) / (2. * N)

# t-SNE definition
def tsne(X, ydim=2, T=1000, l=500, perp=30):
    N = X.shape[0]
    P = p_joint(X, perp)

    Y = []
    y = np.random.normal(loc=0.0, scale=1e-4, size=(N,ydim))
    Y.append(y); Y.append(y)

    for t in range(T):
        Q = q_joint(Y[-1])
        grad = gradient(P, Q, Y[-1])
        y = Y[-1] - l*grad + m(t)*(Y[-1] - Y[-2])
        Y.append(y)
        if t % 10 == 0:
            Q = np.maximum(Q, 1e-12)
        print(t)
    return y

# PCA definition
def pca(high_dimesion_data):
    #find the co-variance matrix which is : A^T * A
    sample_data = high_dimesion_data
    # matrix multiplication using numpy
    covar_matrix = np.matmul(sample_data.T , sample_data)
    # this code generates only the top 2 (782 and 783)(index) eigenvalues.
    _, vectors = eigh(covar_matrix, eigvals=(2,3))
    
    return np.matmul(vectors.T, sample_data.T).T


# Main
modelo = "model2"
route = "A0"
load_t_sne = False
#X, y = load_digits(return_X_y=True)

with open(f'models/{modelo}/latent_test/latent_space_{route}_measured.npy', 'rb') as f:
    X_m = np.squeeze(np.load(f))
with open(f'models/{modelo}/latent_test/latent_space_{route}_interpolated.npy', 'rb') as f:
    X_i = np.squeeze(np.load(f))

scaler = StandardScaler()
X_scaled_m = scaler.fit_transform(X_m)
m_numel = X_scaled_m.shape[0]
X_scaled_i = scaler.fit_transform(X_i)
i_numel = X_scaled_i.shape[0]
X_scaled = np.concatenate((X_scaled_m,X_scaled_i),axis=0)
colours = np.empty((X_scaled.shape[0],3))
colours[0:m_numel,:] = np.tile(np.array([102, 204, 255]),(m_numel,1))
colours[m_numel:m_numel+i_numel,:] = np.tile(np.array([255, 0, 0]),(i_numel,1))
colours = colours/255

res = pca(X_scaled)

f, (ax1,ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.scatter(res[:, 0], res[:, 1], s=20, c=colours) # c=y para colorear los puntos
ax1.title.set_text('PCA')

if load_t_sne:
    with open(f'models/{modelo}/t_sne.npy', 'rb') as f:
        res = np.load(f)
else:
    res = tsne(X_scaled, T=10000, l=200, perp=40) # Me guardo res en el fichero t_sne.npy (para evitar recalcularlo)
    with open(f'models/{modelo}/t_sne.npy', 'wb') as f:
        np.save(f,res)

ax2.scatter(res[:, 0], res[:, 1], s=20, c=colours) # c=y para colorear los puntos
ax2.title.set_text('t-SNE')
plt.show()