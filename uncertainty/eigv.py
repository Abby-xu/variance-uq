import generate_responses
from models import gpt
import argparse
import torch
from typing import List
import models
import numpy as np
from collections import Counter

def get_L_mat(W, symmetric=True):
    # compute the degreee matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
    if symmetric:
        L = np.linalg.inv(np.sqrt(D)) @ (D - W) @ np.linalg.inv(np.sqrt(D))
    else:
        raise NotImplementedError()
        # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
        L = np.linalg.inv(D) @ (D - W)
    return L.copy()

def get_eig(L, thres=None, eps=None):
    # This function assumes L is symmetric
    # compute the eigenvalues and eigenvectors of the laplacian matrix
    if eps is not None:
        L = (1-eps) * L + eps * np.eye(len(L))
    eigvals, eigvecs = np.linalg.eigh(L)

    #eigvals, eigvecs = np.linalg.eig(L)
    #assert np.max(np.abs(eigvals.imag)) < 1e-5
    #eigvals = eigvals.real
    #idx = eigvals.argsort()
    #eigvals = eigvals[idx]
    #eigvecs = eigvecs[:,idx]

    if thres is not None:
        keep_mask = eigvals < thres
        eigvals, eigvecs = eigvals[keep_mask], eigvecs[:, keep_mask]
    return eigvals, eigvecs

def calculate_ecc(question, responses, wrapper, mode='entailment'):
    mat_results = wrapper.create_sim_mat_batched(question, responses)
    mat, mapping = mat_results['sim_mat'], mat_results['mapping']
    if mode == 'entailment':
        mat = mat[:, :, 2]
    elif mode == 'contradiction':
        mat = 1 - mat[:, :, 0]

    m = mat.shape[0]
    if m == 1:
        return 0
    num_responses = len(responses)
    print(mapping)
    
    mat = (mat + mat.T) / 2
    # print(mat.shape)
    # print(mat)
    mat += np.eye(mat.shape[0])
    # mat to numpy
    mat = mat.cpu().numpy()
    L = get_L_mat(mat)
    eigvals, eigvecs = get_eig(L)
    print(eigvecs.shape)
    response_mat = np.zeros((num_responses, m))
    print(response_mat.shape)
    for i, response in enumerate(responses):
        j = mapping[i] # TODO: is this how mapping works?
        response_vec = np.zeros(L.shape[0])
        for vec_idx, vec in enumerate(eigvecs):
            response_vec[vec_idx] = vec[j]
        response_mat[i] = response_vec
    response_mat = response_mat.T
    cov_mat = np.cov(response_mat)
    uncertainty = np.trace(cov_mat)
    return uncertainty
    

def calculate_eigv(question, responses, wrapper, mode='entailment'):
    mat_results = wrapper.create_sim_mat_batched(question, responses)
    mat, mapping = mat_results['sim_mat'], mat_results['mapping']
    if mode == 'entailment':
        mat = mat[:, :, 2]
    elif mode == 'contradiction':
        mat = 1 - mat[:, :, 0]

    
    mat = (mat + mat.T) / 2
    # print(mat.shape)
    # print(mat)
    mat += np.eye(mat.shape[0])
    # mat to numpy
    mat = mat.cpu().numpy()
    L = get_L_mat(mat)
    eigvals, eigvecs = get_eig(L)
    eigvals = 1-eigvals
    # print(eigvals)
    # print(eigvals.clip(0))
    uncertainty = eigvals.clip(0).sum()
    # print(uncertainty)
    # turn uncertainty into a torch val
    uncertainty = torch.tensor(uncertainty)

    return uncertainty

    

