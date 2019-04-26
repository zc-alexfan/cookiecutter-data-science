'''
Unsort a list L using its sorting indices obtained in its sorting step
'''
def unsort(L, sort_idx): 
    LL = zip(sort_idx, L)
    LL = sorted(LL, key=lambda x: x[0])
    _, L = zip(*LL)
    return list(L)
