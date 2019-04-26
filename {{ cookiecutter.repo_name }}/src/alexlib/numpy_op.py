
def sort_2d_np(edges):
    """
    edges: (m x 2)
    Sort ascendngly according to each column.
    Example: 
    edges = array(
              [[9, 8],
               [3, 6],
               [1, 3],
               [8, 6],
               [3, 2],
               [9, 1],
               [5, 1]]
             )
    
    sorted_edges = array(
              [[1, 3],
               [3, 2],
               [3, 6],
               [5, 1],
               [8, 6],
               [9, 1],
               [9, 8]]
             )
    """
    sort_idx = np.arange(len(edges))
    sort_dim1 = edges[:,1].argsort()
    sort_idx = sort_idx[sort_dim1]
    edges = edges[sort_dim1]
    sort_dim0 = edges[:,0].argsort()
    sort_idx = sort_idx[sort_dim0]
    edges = edges[sort_dim0]

    return (sort_idx, edges)



def softmax_np(x):
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 1
    x = np.exp(x)
    return x/x.sum()

def l1_normalize_np(x):
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 1
    return x/x.sum()



def diag_mask_np(n):
    return np.array([[i, i] for i in range(n)])

