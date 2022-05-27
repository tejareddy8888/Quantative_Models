import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import osqp
import scipy.sparse as sparse

def solve_qp(labels, features, slack):
    m = osqp.OSQP()
    n1 = features.shape[1] + 1
    n2 = labels.shape[0]
    P = sparse.block_diag((sparse.eye(n1), sparse.csr_matrix(np.zeros((n2, n2)))))
    q = np.vstack([np.zeros((n1, 1)), slack * np.ones((n2, 1))])

    a11 = sparse.csr_matrix(np.expand_dims(labels, 1) * np.concatenate((features, np.ones((inputs.shape[0], 1))), axis=1))
    a12 = sparse.eye(n2)
    a21 = sparse.csr_matrix(np.zeros((n2, n1)))
    a22 = sparse.eye(n2)
    A = sparse.vstack([sparse.hstack([a11, a12]), sparse.hstack([a21, a22])])
    l = np.vstack([np.ones((n2, 1)), np.zeros((n2, 1))])
    m.setup(P=P, q=q, A=A, l=l)
    results = m.solve()
    return results.x[:n1], results.y[:n2]

def compute_rbf(data, centers):
    tmp = np.exp(-np.sum((data-np.expand_dims(centers, axis=1))**2, axis=2) / 0.2)
    return np.transpose(tmp)

def equiv_kernel(test_data, support_inputs, support_labels, centers):
    phi = compute_rbf(support_inputs, centers)
    V = np.linalg.inv(np.matmul(np.transpose(phi), phi))
    tmp = np.matmul(np.matmul(phi * np.expand_dims(support_labels, 1), V), np.transpose(compute_rbf(test_data, centers)))
    return np.sum(tmp, axis=0)

#Read the data
df = pd.read_excel('toy_data.xls')

# Assign in and output
inputs = df[['x1', 'x2']].values
labels = df['y'].values

# Place RBF centers (requires prior knowledge)
centers = np.array([[0.23, 0.76], [0.80, 0.83], [0.56, 0.51], [0.26, 0.23], [0.83, 0.16]])
features = compute_rbf(inputs, centers)

x, y = solve_qp(labels, features, 100000)
ix = np.argsort(y)[:13]
support_inputs = inputs[ix,:]
support_labels = labels[ix]

plt.figure()
plt.plot(labels)
# eval model
pred_eval = equiv_kernel(inputs, support_inputs, support_labels, centers)
plt.plot(pred_eval)
plt.legend(['ground truth', 'model' ])

plt.figure()
plt.scatter(inputs[:49,0], inputs[:49, 1], marker='.', c='b')
plt.scatter(inputs[49:,0], inputs[49:, 1], marker='.', c='r')
plt.scatter(support_inputs[:,0], support_inputs[:,1], marker='o', c='k')
plt.scatter(centers[:,0], centers[:,1], marker='*', c='g')

# test model
[x0, x1] = np.meshgrid(np.arange(0, 1, 0.05), np.arange(0, 1, 0.05))
test_data = np.transpose(np.array([x0, x1]).reshape(2, x0.shape[0] ** 2))
pred_test = equiv_kernel(test_data, support_inputs, support_labels, centers)
x2 = pred_test.reshape(x0.shape[0], x0.shape[0])
plt.contour(x0, x1, x2)
plt.show()



