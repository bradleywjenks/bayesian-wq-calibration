"""
Functions for simulating hydraulics and water quality using our own solvers.
"""

import networkx as nx
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from pydantic import BaseModel
from typing import Any
from bayesian_wq_calibration.constants import NETWORK_DIR
import json


def hydraulic_solver(wdn, d, h0, C, C_dbv, eta, method='null_space', print_status=False):

    with open(NETWORK_DIR / 'valve_info.json') as f:
        valve_info = json.load(f)

    A12 = wdn.A12
    A10 = wdn.A10
    net_info = wdn.net_info
    link_df = wdn.link_df
    node_df = wdn.node_df
    nt = d.shape[1]
    dbv_links = valve_info['dbv_link']
    dbv_idx = link_df[link_df['link_ID'].isin(dbv_links)].index
    reservoir_nodes = net_info['reservoir_names']
    reservoir_idx = node_df[node_df['node_ID'].isin(reservoir_nodes)].index
    prv_links = valve_info['prv_link']
    prv_dir = valve_info['prv_dir']
    prv_idx = link_df[link_df['link_ID'].isin(prv_links)].index
    A13 = np.zeros((net_info['np'], len(prv_idx)))

    for col, idx in enumerate(prv_idx):
        A13[idx, col] = 1
    A13 = sp.csr_matrix(A13)

    n_exp = link_df['n_exp'].astype(float).to_numpy().reshape(-1, 1)
    K = np.zeros((net_info['np'], nt))

    for idx, row in link_df.iterrows():
        if row['link_type'] == 'pipe':
            K[idx, :] = friction_loss(net_info, row, C[idx])

        elif row['link_type'] == 'valve':
            K[idx, :] = local_loss(row, C[idx])

    # assign time-varying loss coefficients at DBV links
    for idx in range(C_dbv.shape[0]):
        for t in range(C_dbv.shape[1]):
            link_idx = dbv_idx[idx]
            K[link_idx, t] = local_loss(row, C_dbv[idx, t])

    tol = 1e-5
    kmax = 50
    tol_A11 = 1e-5

    q = np.zeros((net_info['np'], nt))
    h = np.zeros((net_info['nn'], nt))

    if method == 'null_space':
        null_data = make_nullspace(A12)
        Z = null_data.Z.tocsr()
        A12_fac = null_data.fac
        A12_T = A12.T
        Z_T = Z.T
        kappa = 1e7


    # run over all time steps
    for t in range(net_info['nt']):
        
        hk = 130 * np.ones((net_info['nn'], 1))
        qk = 0.03 * np.ones((net_info['np'], 1))

        dk = d[:, t].reshape(-1, 1)
        h0k = h0[:, t].reshape(-1, 1)
        etak = eta[:, t].reshape(-1, 1)

        A11_diag = K[:, t].reshape(-1, 1) * (abs(qk) ** (n_exp - 1))
        A11_diag[A11_diag < tol_A11] = tol_A11
        A11 = sp.diags(A11_diag.T, [0])

        if method == 'null_space':
            F_diag = n_exp * A11_diag
            w = A12_fac.solve(dk)
            x = A12 @ w
            x = x.reshape(-1, 1)

        for k in range(kmax):

            if method == 'nr':
                N = sp.diags(n_exp.T, [0]) # matrix N  
                I = sp.eye(net_info['np'], format='csr') # identiy matrix with dimension np x np, allocated as a sparse matrix
                b = np.concatenate([(N - I) @ A11 @ qk - A10 @ h0k - A13 @ etak, dk])
                J = sp.bmat([[N @ A11, A12], [A12.T, sp.csr_matrix((net_info['nn'], net_info['nn']))]], format='csr')

                x = sp.linalg.spsolve(J, b)
                qk = x[:net_info['np']]; qk = qk.reshape(-1, 1)
                hk = x[net_info['np']:net_info['np'] + net_info['nn']];hk = hk.reshape(-1, 1)

                A11_diag = K[:, t].reshape(-1, 1) * (abs(qk) ** (n_exp - 1)) # diagonal elements of matrix A11
                A11_diag[A11_diag < tol_A11] = tol_A11 # replace with small value = tol_A11
                A11 = sp.diags(A11_diag.T, [0]) # matrix A11, allocated as a sparse diagonal matrix

            elif method == 'nr_schur':
                inv_A11_diag = 1 / A11_diag # diagonal elements of the inverse of A11
                inv_A11 = sp.diags(inv_A11_diag.T, [0]) # inverse of A11, allocated as a sparse, diagonal matrix
                inv_N = sp.diags(1/n_exp.T, [0]) # inverse of matrix N
                DD = inv_N @ inv_A11 # matrix inv_N * inv_A11
                b = -A12.T @ inv_N @ (qk + inv_A11 @ (A10 @ h0k + A13 @ etak)) + A12.T @ qk - dk # right-hand side of linear system for finding h^{k+1]
                A = A12.T @ DD @ A12 # Schur complement

                hk = sp.linalg.spsolve(A, b); hk = hk.reshape(-1, 1)

                I = sp.eye(net_info['np'], format='csr') # identiy matrix with dimension np x np, allocated as a sparse matrix
                qk = (I - inv_N) @ qk - DD @ ((A12 @ hk) + (A10 @ h0k))

                A11_diag = K[:, t].reshape(-1, 1) * (abs(qk) ** (n_exp - 1)) # diagonal elements of matrix A11
                A11_diag[A11_diag < tol_A11] = tol_A11 # replace with small value = tol_A11
                A11 = sp.diags(A11_diag.T, [0]) # matrix A11, allocated as a sparse diagonal matrix
                

            ### Null space solver reference: Abraham, E. and Stoianov, I. (2016), "Sparse null space algorithms for hydraulic analysis of large-scale water supply networks.' Journal of Hydraulic Engineering, vol. 142, no. 3. ###
            elif method == 'null_space':
                sigma_max = np.max(F_diag)
                tk = np.maximum((sigma_max / kappa) - F_diag, 0)
                F_diag = F_diag + tk
                X = Z_T @ sp.diags(F_diag.reshape(-1)) @ Z
                b = Z_T @ ((F_diag - A11_diag) * qk - A10 * h0k - A13 @ etak - F_diag * x)
                v = sp.linalg.spsolve(X, b)
                v = v.reshape(-1,1)

                qk_new = x + Z @ v

                b = A12_T @ ((F_diag - A11_diag) * qk - A10 * h0k - A13 @ etak - F_diag * qk_new)
                hk_new = A12_fac.solve(b)
                hk_new = hk_new.reshape(-1, 1)

                hk = hk_new
                qk = qk_new

                A11_diag = K[:, t].reshape(-1, 1) * (abs(qk) ** (n_exp - 1)) # diagonal elements of matrix A11
                A11_diag[A11_diag < tol_A11] = tol_A11 # replace with small value = tol_A11
                A11 = sp.diags(A11_diag.T, [0]) # matrix A11, allocated as a sparse diagonal matrix
                F_diag = n_exp * A11_diag

            else:
                print('Error. No hydraulic solver method was inputted.')

            err = A11 @ qk + A12 @ hk + A10 @ h0k + A13 @ etak
            max_err = np.linalg.norm(err, np.inf)

            # print progress
            if print_status == True:
                print(f"Time step t={t+1}, Iteration k={k}. Maximum energy conservation error is {max_err} m.")

            if max_err < tol:
                break
                
        q[:, t] = qk.T
        h[:, t] = hk.T
    
    return q, h
    


"""
helper functions for hydraulic solver
"""

def friction_loss(net_info, df, C):
    if net_info['headloss'] == 'H-W':
        K = 10.67 * df['length'] * (C ** -df['n_exp']) * (df['diameter'] ** -4.8704)
    else:
        print('Error. Only the Hazen-Williams head loss model is supported.')
        K = [] # insert DW formula here...
    return K

def local_loss(df, C):
    K = (8 / (np.pi ** 2 * 9.81)) * (df['diameter'] ** -4) * C
    return K


class NullData(BaseModel):
    fac: Any
    Z: Any


def make_nullspace(A12):
    n_p, n_n = np.shape(A12)
    n_c = n_p - n_n
    Pt, Rt, T = permute_cotree(A12)

    T1 = sp.tril(T[:n_n, :n_n]).tocsc()
    T2 = T[n_n:n_p, :n_n]
    L21 = sp.csr_matrix(-T2.dot(sp.linalg.spsolve(T1, sp.eye(n_n, format='csc'))))

    Z = Pt.T.dot(sp.hstack((L21, sp.eye(n_c))).T)
    fac = splu(A12.T.dot(A12))

    nulldata = NullData(fac=fac, Z=Z)

    return nulldata


def permute_cotree(A):
    n, m = np.shape(A)
    Pt = sp.eye(n)
    Rt = sp.eye(m)

    for i in range(m):
        K = A[i:n, i:m]
        r = np.argmax(np.sum(np.abs(K), axis=1) == 1)
        c = np.argmax(np.abs(K[r, :]) == 1)

        if r != 0:
            iP = np.arange(n)
            jP = np.arange(n)
            vP = np.ones(n)
            jP[i] = i + r
            jP[i+r] = i
            P = sp.csr_matrix((vP, (iP, jP)), shape=(n, n))
            Pt = P.dot(Pt)
            A = P.dot(A)
        
        if c != 0:
            iR = np.arange(m)
            jR = np.arange(m)
            vR = np.ones(m)
            jR[i] = i + c
            jR[i+c] = i
            R = sp.csr_matrix((vR, (iR, jR)), shape=(m, m))
            Rt = R.dot(Rt)
            A = A.dot(R)

    return Pt, Rt, A