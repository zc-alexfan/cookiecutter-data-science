
def sort_2d_tensor(edges):
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
    edges = edges.cpu().detach().numpy()
    return sort_2d_np(edges)




def all_comb(X, Y): 
    """
    Returns all possible combinations of elements in X and Y.
    X: (n_x, d_x)
    Y: (n_y, d_y)
    Output: Z: (n_x*x_y, d_x+d_y)
    
    Example: 
    X = tensor([[8, 8, 8],
                [7, 5, 9]])
                
    Y = tensor([[3, 8, 7, 7],
                [3, 7, 9, 9],
                [6, 4, 3, 7]])
                
    Z = tensor([[8, 8, 8, 3, 8, 7, 7],
                [8, 8, 8, 3, 7, 9, 9],
                [8, 8, 8, 6, 4, 3, 7],
                [7, 5, 9, 3, 8, 7, 7],
                [7, 5, 9, 3, 7, 9, 9],
                [7, 5, 9, 6, 4, 3, 7]])
    """
    assert len(X.size()) == 2
    assert len(Y.size()) == 2
    X1 = X.unsqueeze(1)
    Y1 = Y.unsqueeze(0)
    X2 = X1.repeat(1,Y.shape[0],1)
    Y2 = Y1.repeat(X.shape[0],1,1)
    Z = torch.cat([X2,Y2],-1)
    Z = Z.view(-1,Z.shape[-1])
    return Z
