import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from matrix_completion import *


def plot_image(A, name):
    plt.imshow(A.T)
    plt.savefig(name)


if __name__ == "__main__":

    argparse = ArgumentParser()
    argparse.add_argument("--m", default=1000, type=int)
    argparse.add_argument("--n", default=100, type=int)
    argparse.add_argument("--k", default=5, type=int)
    argparse.add_argument("--noise", default=1, type=float)
    argparse.add_argument("--mask-prob", default=0.7, type=float)

    args = argparse.parse_args()

    U, V, R = gen_factorization_with_noise(args.m, args.n, args.k, args.noise)
    mask = gen_mask(args.m, args.n, args.mask_prob)

    plot_image(R, "R.png")
    plot_image(mask, "mask.png")
    
    print("== No Matrix Completion")
    print("RMSE:", calc_unobserved_rmse(U, V, R * mask, mask))
    
    
    R_hat = svt_solve_inspired(R * mask, 100, 10, args.k, mask, tau = 50, delta = 0.2 , epsilon=0.1)
    print("== quantum_inspired_svt_solve")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat, "quantum_inspired_svt_solve.png")
    
    R_hat = pmf_solve_inspired(R * mask , 100, 10, args.k, mask, 1e-2)
    print("== pmf_solve_inspired")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat, "pmf_solve.png")

    R_hat = pmf_solve(R * mask , mask, args.k, 1e-2)
    print("== PMF")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat, "pmf_solve.png")

    R_hat = biased_mf_solve(R * mask, mask, args.k, 1e-2)
    print("== BMF")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat, "biased_mf_solve.png")
    
    R_hat = svt_solve(R, mask)
    print("== SVT")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat)
