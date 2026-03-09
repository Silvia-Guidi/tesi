import numpy as np
import networkx as nx

def bayes_factor(residuals, i, j):
    res_i=residuals[:,i]
    res_j=residuals[:,j]
    correlation= np.corrcoef(res_i,res_j)[0,1]
    n=len(res_i)
    bf_removal= np.sqrt(n)*(1-correlation**2)**(n/2)
    return bf_removal

def is_acyclic(matrix):
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    return nx.is_directed_acyclic_graph(G)
