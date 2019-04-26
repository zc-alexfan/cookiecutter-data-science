import torch
import numpy as np

def intersect(ra, rb): 
    assert type(ra) == type(torch.Tensor([]))
    x1, y1, x2, y2 = ra[:4]
    a1, b1, a2, b2 = rb[:4]
    x_overlap = torch.max(torch.Tensor([0])[0], torch.min(x2, a2) - torch.max(x1, a1))
    y_overlap = torch.max(torch.Tensor([0])[0], torch.min(y2, b2) - torch.max(y1, b1))
    return x_overlap*y_overlap

def union(ra, rb, intersect_area): 
    assert type(ra) == type(torch.Tensor([]))
    x1, y1, x2, y2 = ra[:4]
    a1, b1, a2, b2 = rb[:4]
    area_a = (x2-x1)*(y2-y1)
    area_b = (a2-a1)*(b2-b1)
    return area_a+area_b - intersect_area

def union_edges(ra, rb, intersect_area): 
    assert type(ra) == type(torch.Tensor([]))
    assert type(rb) == type(torch.Tensor([]))
    assert ra.size(0) == rb.size(0)
    assert ra.size(1) == rb.size(1)
    assert ra.size(1) == 4


    x1, y1, x2, y2 = torch.chunk(ra, 4, dim=1)
    a1, b1, a2, b2 = torch.chunk(rb, 4, dim=1)

    area_a = (x2-x1)*(y2-y1)
    area_b = (a2-a1)*(b2-b1)
    return area_a+area_b - intersect_area

def get_union_box_tensor(boxes, edges): 
    assert isinstance(boxes, torch.Tensor)
    assert isinstance(edges, torch.LongTensor)
    boxes = boxes.cpu().detach().numpy()
    edges = edges.cpu().detach().numpy()
    return torch.FloatTensor(get_union_box_np(boxes, edges)) 


def get_union_box_np(boxes, edges): 
    assert type(boxes) == type(np.array([]))
    assert type(edges) == type(np.array([]))
    """
    Return the union box of box pairs
    boxes: N x 4 in x1, y1, x2, y2
    edges: M x 3 in from, to
    """
    if len(edges) == 0:
        return np.array([]).reshape(-1, 4)
    
    assert edges.shape[1] == 2
    
    left_box_idx, right_box_idx = edges[:, 0], edges[:, 1]
    left_boxes  = boxes[left_box_idx, :4]
    right_boxes  = boxes[right_box_idx, :4]
    
    points_x = np.concatenate((left_boxes[:, [0, 2]], right_boxes[:, [0, 2]]), axis=1)
    points_y = np.concatenate((left_boxes[:, [1, 3]], right_boxes[:, [1, 3]]), axis=1)
    
    top_left_x = points_x.min(axis=1)
    top_left_y = points_y.min(axis=1)
    bottom_right_x = points_x.max(axis=1)
    bottom_right_y = points_y.max(axis=1)

    union_boxes = np.stack((top_left_x, top_left_y, bottom_right_x, bottom_right_y), axis=1)
    return union_boxes.astype(float)

def intersect_edges(ra, rb): 
    """
    ra: N x 4
    rb: N x 4
    Both in (x1, y1, x2, y2) form
    Output: instersection area of the N pairs in N x 1
    """
    assert type(ra) == type(torch.Tensor([]))
    assert type(rb) == type(torch.Tensor([]))
    assert ra.size(0) == rb.size(0)
    assert ra.size(1) == rb.size(1)
    assert ra.size(1) == 4


    x1, y1, x2, y2 = torch.chunk(ra, 4, dim=1)
    a1, b1, a2, b2 = torch.chunk(rb, 4, dim=1)

    x_overlap = torch.max(torch.Tensor([0])[0], torch.min(x2, a2) - torch.max(x1, a1))
    y_overlap = torch.max(torch.Tensor([0])[0], torch.min(y2, b2) - torch.max(y1, b1))
    intersect_results = x_overlap*y_overlap

    assert intersect_results.size(0) == ra.size(0)
    return intersect_results

def box_iou(ra, rb): 
    assert type(ra) == type(torch.Tensor([]))
    top = intersect(ra, rb)
    bottom = union(ra, rb, top)
    return top/bottom


"""
Compute pair-iou overlaps between two pairs of bboxes (i.e. 4 boxes)
"""
def pair_iou(ru, rv, rp, rq): 
    assert type(ru) == type(torch.Tensor([]))
    up_intersect = intersect(ru, rp)
    vq_intersect = intersect(rv, rq)
    top = up_intersect + vq_intersect
    bottom = union(ru, rp, up_intersect) + union(rv, rq, vq_intersect)
    return top/bottom

def edge_iou(edges):
    assert type(edges) == type(torch.Tensor([]))
    assert edges.size(1) == 16 # 4 * 2 * 2
    assert edges.size(0) > 0

    ru, rv, rp, rq = torch.chunk(edges, 4, dim=1)
    assert ru.size(1) == 4
    up_intersect = intersect_edges(ru, rp)
    vq_intersect = intersect_edges(rv, rq)
    top = up_intersect + vq_intersect
    bottom = union_edges(ru, rp, up_intersect) + union_edges(rv, rq, vq_intersect)

    ious = top/bottom
    return ious



