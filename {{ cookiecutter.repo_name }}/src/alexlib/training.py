def set_lr(optimizer, lr):
    assert isinstance(lr, float)
    for g in optimizer.param_groups:
        g['lr'] = lr


def reduce_lr(optimizer, gamma):
    assert isinstance(gamma, float)
    for g in optimizer.param_groups:
        g['lr'] *= gamma 