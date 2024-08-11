import cv2
import numpy as np
import torch
import torch.nn.functional as F

def tensor_dilate(bin_img, ksize=21):
    # mask torch.Tensor 1, H, W
    # print(bin_img.shape)
    bin_img = bin_img.clone()[None]
    # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k
    # print(patches.shape)

    # 取每个 patch 中最小的值，i.e., 0
    dilated, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
    # print(dilated.shape)
    return dilated


def extract_random_expand_neg_mask_points(mask, n_points_to_select):
    neg_mask = tensor_dilate(mask) - mask
    # print()
    # neg_mask_np = neg_mask.squeeze(0).permute(1,2,0).cpu().numpy() * 255
    # cv2.imwrite("test.png", neg_mask_np)

    # raise

    if neg_mask.sum() == 0:
        print(f"Warning: neg_mask.sum() == 0 in extract_random_mask_points")
        return torch.zeros((n_points_to_select, 2))
    
    mask_pixels = neg_mask.squeeze(dim=0).permute(1,2,0).nonzero().float()
    assert len(mask_pixels) > 0
    if len(mask_pixels) < n_points_to_select:
        selected_points = mask_pixels.repeat(n_points_to_select // len(mask_pixels) + 1, 1)[:n_points_to_select]
    else:
        selected_points = mask_pixels[torch.randperm(len(mask_pixels))[:n_points_to_select]]

    selected_points = selected_points.flip(1)  # Change from (y, x) to (x, y)
    
    assert selected_points.shape == (n_points_to_select, 3)
    return selected_points

def extract_uniform_expend_neg_mask_points(mask, n_points_to_select):
    neg_mask = tensor_dilate(mask) - mask

    # print(neg_mask.shape)

    if neg_mask.sum() == 0:
        print(f"Warning: neg_mask.sum() == 0 in extract_random_mask_points")
        return torch.zeros((n_points_to_select, 2))
    
    mask_pixels = neg_mask.squeeze(dim=0).permute(1,2,0).nonzero().float()
    assert len(mask_pixels) > 0
    if len(mask_pixels) < n_points_to_select:
        selected_points = mask_pixels.repeat(n_points_to_select // len(mask_pixels) + 1, 1)[:n_points_to_select]
    else:
        n_select_interival = mask_pixels.shape[0] // n_points_to_select // 2
        # selected_points = mask_pixels[0:len(mask_pixels):n_select_interival]
        selected_points = mask_pixels[0:len(mask_pixels)//4:n_select_interival]
        selected_points = mask_pixels[len(mask_pixels)//4*3:len(mask_pixels):n_select_interival]

    selected_points = selected_points.flip(1)  # Change from (y, x) to (x, y)
    selected_points = selected_points[:n_points_to_select]

    # print(selected_points.shape)
    
    assert selected_points.shape == (n_points_to_select, 3)
    return selected_points


def extract_random_whole_neg_mask_points(mask, n_points_to_select):
    neg_mask = torch.ones_like(mask) - tensor_dilate(mask).squeeze(0)
    # print(tensor_dilate(mask).shape)

    if neg_mask.sum() == 0:
        print(f"Warning: neg_mask.sum() == 0 in extract_random_mask_points")
        return torch.zeros((n_points_to_select, 2))
    
    mask_pixels = neg_mask.permute(1,2,0).nonzero().float()
    assert len(mask_pixels) > 0
    if len(mask_pixels) < n_points_to_select:
        selected_points = mask_pixels.repeat(n_points_to_select // len(mask_pixels) + 1, 1)[:n_points_to_select]
    else:
        selected_points = mask_pixels[torch.randperm(len(mask_pixels))[:n_points_to_select]]

    selected_points = selected_points.flip(1)  # Change from (y, x) to (x, y)
    
    assert selected_points.shape == (n_points_to_select, 3)
    return selected_points


def extract_uniform_whole_neg_mask_points(mask, n_points_to_select):
    neg_mask = torch.ones_like(mask) - tensor_dilate(mask).squeeze(0)
    # print(tensor_dilate(mask).shape)

    if neg_mask.sum() == 0:
        print(f"Warning: neg_mask.sum() == 0 in extract_random_mask_points")
        return torch.zeros((n_points_to_select, 2))
    
    mask_pixels = neg_mask.permute(1,2,0).nonzero().float()
    assert len(mask_pixels) > 0
    if len(mask_pixels) < n_points_to_select:
        selected_points = mask_pixels.repeat(n_points_to_select // len(mask_pixels) + 1, 1)[:n_points_to_select]
    else:
        n_select_interival = mask_pixels.shape[0] // n_points_to_select // 2
        # selected_points = mask_pixels[0:len(mask_pixels):n_select_interival]
        selected_points = mask_pixels[0:len(mask_pixels)//2:n_select_interival]
        selected_points = mask_pixels[len(mask_pixels)//2:len(mask_pixels):n_select_interival]

    selected_points = selected_points.flip(1)  # Change from (y, x) to (x, y)
    selected_points = selected_points[:n_points_to_select]
    # print(selected_points.shape)
    assert selected_points.shape == (n_points_to_select, 3)
    return selected_points


def extract_random_mask_points(mask, n_points_to_select):
    # mask torch.Tensor 1, H, W
    # point torch.Tensor N_point, 3

    if mask.sum() == 0:
        print(f"Warning: mask.sum() == 0 in extract_random_mask_points")
        return torch.zeros((n_points_to_select, 2))
    
    mask_pixels = mask.permute(1,2,0).nonzero().float()
    assert len(mask_pixels) > 0
    if len(mask_pixels) < n_points_to_select:
        selected_points = mask_pixels.repeat(n_points_to_select // len(mask_pixels) + 1, 1)[:n_points_to_select]
    else:
        selected_points = mask_pixels[torch.randperm(len(mask_pixels))[:n_points_to_select]]

    selected_points = selected_points.flip(1)  # Change from (y, x) to (x, y)
    
    assert selected_points.shape == (n_points_to_select, 3)
    return selected_points

def extract_uniform_mask_points(mask, n_points_to_select):
    if mask.sum() == 0:
        print(f"Warning: mask.sum() == 0 in extract_random_mask_points")
        return torch.zeros((n_points_to_select, 2))
    
    mask_pixels = mask.permute(1,2,0).nonzero().float() # 非零未知的索引

    assert len(mask_pixels) > 0
    if len(mask_pixels) < n_points_to_select:
        selected_points = mask_pixels.repeat(n_points_to_select // len(mask_pixels) + 1, 1)[:n_points_to_select]
    else:
        n_select_interival = mask_pixels.shape[0] // n_points_to_select // 2
        selected_points = mask_pixels[mask_pixels.shape[0]//4:mask_pixels.shape[0]//4*3:n_select_interival]
        # selected_points = mask_pixels[0:len(mask_pixels):n_select_interival]

    selected_points = selected_points.flip(1)  # Change from (y, x) to (x, y)

    return selected_points


def extract_center_mask_points(mask):
    if mask.sum() == 0:
        print(f"Warning: mask.sum() == 0 in extract_random_mask_points")
        return torch.zeros((1, 2))
    mask_pixels = mask.permute(1,2,0).nonzero().float() # 非零未知的索引
    selected_points = mask_pixels[len(mask_pixels)//2][None]
    # print(selected_points.shape)
    selected_points = selected_points.flip(1)
    return selected_points


def Cross_Points(query, num):
    # query 1,4 numpy
    pos_query = torch.from_numpy(np.array([query[0,0]+query[0,2]//2, query[0,1]+query[0,3]//2]))
    pos_query_points = torch.zeros((num,3)).cuda()
    pos_query_points[0,1] = pos_query[0]
    pos_query_points[0,2] = pos_query[1]
    
    idx = 0
    count = 1
    for i in range(1, num):
        if idx == 0:
            pos_query_points[i,1] = pos_query[0] - 25 * count
            # pos_query_points[i,1] = pos_query[0] - 10 * count
            pos_query_points[i,2] = pos_query[1]
            idx += 1
        elif idx == 1:
            pos_query_points[2,1] = pos_query[0] + 25 * count
            # pos_query_points[2,1] = pos_query[0] + 10 * count
            pos_query_points[2,2] = pos_query[1]
            idx += 1
        elif idx == 2:
            pos_query_points[3,1] = pos_query[0] 
            # pos_query_points[3,2] = pos_query[1] - 5 * count
            pos_query_points[3,2] = pos_query[1] - 10 * count
            idx += 1
        elif idx == 3:
            pos_query_points[4,1] = pos_query[0]
            # pos_query_points[4,2] = pos_query[1] + 5 * count
            pos_query_points[4,2] = pos_query[1] + 10 * count
            idx = 0
            count += 1
            
    #     if i % 
    # pos_query_points[1,1] = pos_query[0] - 25
    # pos_query_points[1,2] = pos_query[1]
    # pos_query_points[2,1] = pos_query[0] + 25
    # pos_query_points[2,2] = pos_query[1]
    # pos_query_points[3,1] = pos_query[0] 
    # pos_query_points[3,2] = pos_query[1] - 10
    # pos_query_points[4,1] = pos_query[0]
    # pos_query_points[4,2] = pos_query[1] + 10
    return pos_query_points

def Single_Point(query):
    pos_query = torch.from_numpy(np.array([query[0,0]+query[0,2]//2, query[0,1]+query[0,3]//2]))
    pos_query_points = torch.zeros((1,3)).cuda()
    pos_query_points[0,1] = pos_query[0]
    pos_query_points[0,2] = pos_query[1]
    return pos_query_points

# def Random_Point(query):
#     min_x = query[0, 0]
#     min_y = query[0, 1]
#     max_x = query[0, 0] + query[0, 2]
#     max_y = query[0, 1] + query[0, 3]
    
#     pos_query = torch.from_numpy(np.array([query[0,0]+query[0,2]//2, query[0,1]+query[0,3]//2]))
#     pos_query_points = torch.zeros((5,3)).cuda()
#     pos_query_points[0,1] = pos_query[0]
#     pos_query_points[0,2] = pos_query[1]
    
#     for i in range(1, 4):
#         pos_query_points[i,1] = np.random.randint(min_x, max_x)
#         pos_query_points[i,2] = np.random.randint(min_y, max_y)
        
#     return pos_query_points

def Random_Point(mask, num):
    mask_torch = torch.from_numpy(mask).float()
    mask_pixels = mask_torch.nonzero().float()
    # print(mask_pixels.shape)
    idx = np.random.randint(0, mask_pixels.shape[0], num)
    # print(idx) 
    pos_query_points = torch.zeros((num,3)).cuda()
    for i in range(num):
        pos_query_points[i,1] = mask_pixels[idx[i], 1]
        pos_query_points[i,2] = mask_pixels[idx[i], 0]
        
    return pos_query_points

def K_Medoids_Point(mask, num):
    from sklearn_extra.cluster import KMedoids
    mask_torch = torch.from_numpy(mask).float()
    mask_pixels = mask_torch.nonzero().float()

    selected_points = KMedoids(n_clusters=num).fit(mask_pixels).cluster_centers_

    pos_query_points = torch.zeros((num,3)).cuda()
    for i in range(num):
        pos_query_points[i,1] = selected_points[i, 1]
        pos_query_points[i,2] = selected_points[i, 0]
    
    return pos_query_points
    

# def get_random_point(query):
#     min_x = query[0, 0]
#     min_y = query[0, 1]
#     max_x = query[0, 0] + query[0, 2]
#     max_y = query[0, 1] + query[0, 3]
#     pos_query_points = torch.zeros((1,2)).cuda()
#     pos_query_points[0,0] = np.random.randint(min_x, max_x)
#     pos_query_points[0,1] = np.random.randint(min_y, max_y)
#     return pos_query_points
    

if __name__=="__main__":
    image = torch.zeros((1, 1024, 1024))
    image[0, 512-33:512+33, 512-32:512+32] = 1
    image[0, 300-32:300+32, 300-32:300+32] = 1
    image[0, 900-32:900+32, 900-32:900+32] = 1

    tensor_dilate(image)

