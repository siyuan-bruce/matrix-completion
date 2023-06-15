from __future__ import division
import numpy as np
import logging

from sklearn.utils.extmath import randomized_svd, svd_flip
from scipy.sparse.linalg import svds
from numpy.linalg import norm

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import time
import os


def variance_scaled_ls_probs(m, n, A):
    # populates array with the row-norms squared of matrix A
    row_norms = np.zeros(m)
    for i in range(m):
        row_norms[i] = np.abs(la.norm(A[i, :]))**2

    # Frobenius norm of A
    A_Frobenius = np.sqrt(np.sum(row_norms))

    LS_prob_rows = np.zeros(m)

    # normalized length-square row probability distribution
    for i in range(m):
        LS_prob_rows[i] = row_norms[i] / A_Frobenius**2

    LS_prob_columns = np.zeros((m, n))

    # populates array with length-square column probability distributions
    # LS_prob_columns[i]: LS probability distribution for selecting columns from row A[i]
    for i in range(m):
        LS_prob_columns[i, :] = [np.abs(k)**2 / row_norms[i] for k in A[i, :]]

    # New part: compute variances and adjust the probabilities
    row_vars = np.var(A, axis=1)  # row variances
    col_vars = np.var(A, axis=0)  # column variances

    row_adj = 1 / (1 + row_vars)  # adjust row probabilities - lower for high variance
    col_adj = 1 / (1 + col_vars)  # adjust column probabilities - lower for high variance

    LS_prob_rows *= row_adj  # adjust row probabilities
    LS_prob_rows /= np.sum(LS_prob_rows)  # renormalize

    for i in range(m):
        LS_prob_columns[i, :] *= col_adj  # adjust column probabilities
        LS_prob_columns[i, :] /= np.sum(LS_prob_columns[i, :])  # renormalize each row

    return row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius



def ls_probs(m, n, A):

    r"""Function generating the length-squared (LS) probability distributions for sampling matrix A.

    Args:
        m (int): number of rows of matrix A
        n (int): row n of columns of matrix A
        A (array[complex]): most general case is a rectangular complex matrix

    Returns:
        tuple: Tuple containing the row-norms, LS probability distributions for rows and columns,
        and Frobenius norm.
    """

    # populates array with the row-norms squared of matrix A
    row_norms = np.zeros(m)
    for i in range(m):
        row_norms[i] = np.abs(la.norm(A[i, :]))**2

    # Frobenius norm of A
    A_Frobenius = np.sqrt(np.sum(row_norms))

    LS_prob_rows = np.zeros(m)

    # normalized length-square row probability distribution
    for i in range(m):
        LS_prob_rows[i] = row_norms[i] / A_Frobenius**2

    LS_prob_columns = np.zeros((m, n))

    # populates array with length-square column probability distributions
    # LS_prob_columns[i]: LS probability distribution for selecting columns from row A[i]
    for i in range(m):
        LS_prob_columns[i, :] = [np.abs(k)**2 / row_norms[i] for k in A[i, :]]

    return row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius


def sample_C(A, m, n, r, c, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius):

    r"""Function used to generate matrix C by performing LS sampling of rows and columns of matrix A.

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        m (int): number of rows of matrix A
        n (int): number of columns of matrix A
        r (int): number of sampled rows
        c (int): number of sampled columns
        row_norms (array[float]): norm of the rows of matrix A
        LS_prob_rows (array[float]): row LS probability distribution of matrix A
        LS_prob_columns (array[float]): column LS probability distribution of matrix A
        A_Frobenius (float): Frobenius norm of matrix A

    Returns:
        tuple: Tuple containing the singular values (sigma), left- (w) and right-singular vectors (vh) of matrix C,
        the sampled rows (rows), the column LS prob. distribution (LS_prob_columns_R) of matrix R and split running
        times for the FKV algorithm.
    """

    tic = time.time()
    # sample row indices from row length_square distribution
    rows = np.random.choice(m, r, replace=True, p=LS_prob_rows)

    columns = np.zeros(c, dtype=int)
    # sample column indices
    for j in range(c):
        # sample row index uniformly at random
        i = np.random.choice(rows, replace=True)
        # sample column from length-square distribution of row A[i]
        columns[j] = np.random.choice(n, 1, p=LS_prob_columns[i])

    toc = time.time()
    rt_sampling_C = toc - tic

    # building the lenght-squared distribution to sample columns from matrix R
    R_row = np.zeros(n)
    LS_prob_columns_R = np.zeros((r, n))

    tic = time.time()
    # creates empty array for R and C matrices. We treat R as r x c here, since we only need columns later
    R_C = np.zeros((r, c))
    C = np.zeros((r, c))

    # populates array for matrix R with the submatrix of A defined by sampled rows/columns
    for s in range(r):
        for t in range(c):
            R_C[s, t] = A[rows[s], columns[t]]

        # renormalize each row of R
        R_C[s,:] = R_C[s,:] * A_Frobenius / (np.sqrt(r) * np.sqrt(row_norms[rows[s]]))

    # creates empty array of column norms
    column_norms = np.zeros(c)

    # computes column Euclidean norms
    for t in range(c):
        for s in range(r):
            column_norms[t] += np.abs(R_C[s, t])**2

    # renormalize columns of C
    for t in range(c):
        C[:, t] = R_C[:, t] * (A_Frobenius / np.sqrt(column_norms[t])) / np.sqrt(c)

    toc = time.time()
    rt_building_C = toc - tic

    tic = time.time()
    # Computing the SVD of sampled C matrix
    w, sigma, vh = la.svd(C, full_matrices=False)

    toc = time.time()
    rt_svd_C = toc - tic

    return w, rows, sigma, vh, rt_sampling_C, rt_building_C, rt_svd_C


def vl_vector(l, A, r, w, rows, sigma, row_norms, A_Frobenius):

    r""" Function to reconstruct right-singular vector of matrix A

    Args:
        l (int): singular vector index
        A (array[complex]): rectangular, in general, complex matrix
        r (int): number of sampled rows from matrix A
        w (array[complex]): left-singular vectors of matrix C
        rows (array[int]): indices of the r sampled rows of matrix A
        row_norms (array[float]): row norms of matrix A
        A_Frobenius (float): Frobenius norm of matrix A

    Returns:
        array[float]: reconstructed right-singular vector
    """

    n = len(A[1, :])
    v_approx = np.zeros(n)
    # building approximated v^l vector
    factor = A_Frobenius / ( np.sqrt(r) * sigma[l] )
    for s in range(r):
        v_approx[:] += ( A[rows[s], :] / np.sqrt(row_norms[rows[s]]) ) * w[s, l]
    v_approx[:] = v_approx[:] * factor

    return v_approx


def uvl_vector(l, A, r, w, rows, sigma, row_norms, A_Frobenius):

    r""" Function to reconstruct right-singular vector of matrix A

    Args:
        l (int): singular vector index
        A (array[complex]): rectangular, in general, complex matrix
        r (int): number of sampled rows from matrix A
        w (array[complex]): left-singular vectors of matrix C
        rows (array[int]): indices of the r sampled rows of matrix A
        row_norms (array[float]): row norms of matrix A
        A_Frobenius (float): Frobenius norm of matrix A

    Returns:
        tuple: Tuple with arrays containing approximated singular vectors :math: '\bm{u}^l, \bm{v}^l'
    """

    m, n = A.shape
    u_approx = np.zeros(m)
    v_approx = np.zeros(n)
    # building approximated v^l vector
    factor = A_Frobenius / ( np.sqrt(r) * sigma[l] )
    for s in range(r):
        v_approx[:] += ( A[rows[s], :] / np.sqrt(row_norms[rows[s]]) ) * w[s, l]
    v_approx[:] = v_approx[:] * factor

    u_approx = (A @ v_approx) / sigma[l]

    return u_approx, v_approx


def svt_solve_inspired(A, r, c, rank, mask, delta, tau=None, max_iterations=1000, epsilon=1e-5):

    r""" Function to solve the the linear system of equations :math:'A \bm{x} = b' using FKV algorithm
    and a direct calculation of the coefficients :math: '\lambda_l' and solution vector :math: '\bm{x}'

    Args:
        A (array[complex]): rectangular, in general, complex matrix
        r (int): number of sampled rows from matrix A
        c (int): number of sampled columns from matrix A
        rank (int): rank of matrix A

    Returns:
        array[float]: array containing the components of the solution vector :math: '\bm{x}'
    """
    logger = logging.getLogger(__name__)
    Y = mask * A
    
    m_rows, n_cols = np.shape(A)
    
    if not tau:
        tau = 5 * (m_rows + n_cols) / 2

    # 1- Generating LS probability distributions used to sample rows and columns indices of matrix A
    tic = time.time()

    LS = ls_probs(m_rows, n_cols, Y)
    
    
    # save reconstruction error for drawing
    rec_errors = []

    for k in range(max_iterations):
        
        toc = time.time()
        
        rt_ls_prob = toc - tic
        
        if k == 0:
            X = np.zeros_like(A)
        else:
            # 2- Building matrix C by sampling "r" rows and "c" columns from matrix A and computing SVD of matrix C
            svd_C = sample_C(Y, m_rows, n_cols, r, c, *LS[0:4])
            w = svd_C[0]
            sigma = svd_C[2]
            
            ul_approx = np.zeros((m_rows, rank))
            vl_approx = np.zeros((n_cols, rank))
            for l in range(rank):
                ul_approx[:, l], vl_approx[:, l] = uvl_vector(l, Y, r, w, svd_C[1], sigma, LS[0], LS[3])
            
            shrink_S = np.maximum(sigma - tau, 0)
            r_previous = np.count_nonzero(shrink_S)
            
            # Apply regularization to singular values
            sigma = sigma / (1 + delta)
    
            diag_shrink_S = np.diag(shrink_S[:rank])
            
            print(diag_shrink_S)
            
            X_new = np.linalg.multi_dot([ul_approx, diag_shrink_S, vl_approx.T])
                
            X = 0.9 * X + 0.1 * X_new
            
            Y = delta * mask * (Y - X)
        
        recon_error = np.linalg.norm(mask * (X - A)) / np.linalg.norm(mask * A)
        rec_errors.append(recon_error)
        if k % 10 == 0:
            logger.info("Iteration: %i; Rel error: %.4f" % (k + 1, recon_error))
    
        if recon_error < epsilon:
            break
        
    # draw reconstruction error with iterations
    plt.figure()
    plt.plot(rec_errors)
    plt.xlabel('Iterations')
    plt.ylabel('Reconstruction error')
    plt.savefig('reconstruction_error.png')

    return X
        
