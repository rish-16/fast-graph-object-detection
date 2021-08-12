import numpy as np
from einops import rearrange
import torch
from pprint import pprint

def display(arr):
    header = "{:^}\t".format("")
    for i in range(len(arr)):
        header += "{:^}\t".format(i)
    print (header)
    content = ""
    for i, entry in enumerate(arr, 0):
        temp = ""
        for j in entry:
            temp += "{:^}\t".format(j)
        content += "{:^}\t{}\n".format(i, temp)
        
    print (content)

def create_adjacency_matrix(img):
    '''
    An image has D = HxW pixels. 
    Each pixel is a node.
    A pixel's immediate neighbourhood is a the Moore Neighbourhood
    Adjacency Matrix will be DxD
    '''
    def get_neighbourhood(i, j):
        return [
            [i, j],
            [i+1, j],
            [i-1, j],
            [i, j+1],
            [i, j-1],
            [i+1, j+1],
            [i-1, j+1],
            [i+1, j-1],
            [i-1, j-1],
        ]
        
    h, w, c = img.shape
    matrix = np.array([np.array([0 for _ in range(h*w)]) for _ in range(h*w)])
    template = np.array([(i) for i in range(h*w)])
    template = template.reshape([h, w])
    
    def prune_neighbours(arr, h, w):
        def checker(i, j):
            return True if (i >= 0 and i < h and j >= 0 and j < w) else False
        
        return list(filter(lambda x : checker(x[0], x[1]), arr))

    for i in range(h):
        for j in range(w):
            coord = [i, j]
            source_index = template[i][j]
            neighbours = get_neighbourhood(i, j)
            neighbours = prune_neighbours(neighbours, h, w)
            for rec in neighbours:
                target_idx = template[rec[0]][rec[1]]
                matrix[source_index][target_idx] = 1
                
    return matrix