import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import cv2
from torchvision.transforms import Resize

def get_video_list(path):
    video_list = [os.path.join(path,name) for name in os.listdir(path)]
    return video_list

def get_frame_list_ssig(path):
    frame_list = []
    files = os.listdir(path)
    video_name = path.split('/')[-1]
    import re
    pattern = r'\d+'
    # print(re.findall(pattern, video_name))
    # raise
    video_number = int(re.findall(pattern, video_name)[0])
    frame_name = f'Track{video_number}'

    num_frames = 0
    for f in files:
        if '.png' in f:
            num_frames += 1
    
    for i in range(num_frames):
        frame_list.append(os.path.join(path, frame_name+f"[{(i+1):02d}].png"))
    return frame_list

def get_frame_list(path):
    frame_list = []
    files = os.listdir(path)
    video_name = path.split('/')[-1]

    num_frames = 0
    for f in files:
        if '.png' in f:
            num_frames += 1
    
    for i in range(num_frames):
        frame_list.append(os.path.join(path, video_name+f"[{(i+1):02d}].png"))
    return frame_list

def get_annot(img_path):
    path = img_path.replace("png", "txt")
    with open(path, 'r') as f:
        data = f.read()
    
    lines = data.replace('\t', '').replace('-', '').split('\n')
    for line in lines:
        line_split = line.split(':')
        prop = line_split[0].strip()
        if prop == "position_plate":
            data = line_split[1].strip()
            data = data.split(" ")
            data = np.array(data, dtype=np.float32).reshape((1, 4))
    return data

def get_annot_ssig(img_path):
    path = img_path.replace("png", "txt")
    with open(path, 'r') as f:
        data = f.read()
    
    lines = data.replace('\t', '').replace('-', '').split('\n')
    for line in lines:
        line_split = line.split(':')
        # print(len(line_split))
        prop = line_split[0].strip()
        if prop == "position_plate" and len(line_split) == 2:
            data = line_split[1].strip()
            data = data.split(" ")
            data = np.array(data, dtype=np.float32).reshape((1, 4))
    return data


def path2image_np(path):
    image_np = np.array(Image.open(path))
    image_torch = transforms.ToTensor()(image_np)[None]
    return image_torch

def img2video(img1, img2):
    b, c, h, w = img1.shape
    video = torch.cat([img1[None], img2[None]], dim=1)
    return video

def get_mask(img, annot):
    B, C, H, W = img.shape
    mask = torch.zeros((H, W))
    mask[int(annot[:,1]):int(annot[:,1]+annot[:,3]),int(annot[:,0]):int(annot[:,0]+annot[:,2])] = 1
    mask = mask.long()[None, None]
    # print(mask.shape)
    return mask

def get_plate_number(path):
    annot_path = path.replace("png", "txt")
    with open(annot_path, 'r') as f:
        data = f.read()
    lines = data.replace('\t', '').split('\n')
    for line in lines:
        line_split = line.split(':')
        prop = line_split[0].strip()
        data = line_split[1].strip()
        if prop == "plate":
            plate = data.strip().replace("-","").upper()
    return plate

def get_plate_number_ssig(path):
    annot_path = path.replace("png", "txt")
    with open(annot_path, 'r') as f:
        data = f.read()
    lines = data.replace('\t', '').split('\n')
    for line in lines:
        line_split = line.split(':')
        prop = line_split[0].strip()
        if prop == "text":
            data = line_split[1].strip()
            plate = data.strip().replace("-","").upper()
    return plate

def frames2video_ssig(frame_list):
    frames = []
    querys = []
    annots = []
    plates = []
    for i in range(len(frame_list)):
        image_torch = path2image_np(frame_list[i])  # B, C, H, W
        annot = get_annot_ssig(frame_list[i])
        # annot[:,2] = annot[:,2] + annot[:,0]
        # annot[:,3] = annot[:,3] + annot[:,1]
        annots.append(annot)
        plate = get_plate_number_ssig(frame_list[i])
        plates.append(plate)
        mask_torch = get_mask(image_torch, annot) # B, 1, H, W

        if i != 0 and image_torch.shape != frames[0].shape:
            torch_resize = Resize([frames[-1].shape[2], frames[-1].shape[3]])
            image_torch = torch_resize(image_torch)
            mask_torch = torch_resize(mask_torch)

        frames.append(image_torch)
        querys.append(mask_torch)
    
    video = torch.cat(frames, dim=0)[None] # B, T, C, H, W
    querys = torch.cat(querys, dim=0) # T, C, H, W
    return video, querys, annots, plates

def frames2video(frame_list):
    frames = []
    querys = []
    annots = []
    plates = []
    for i in range(len(frame_list)):
        image_torch = path2image_np(frame_list[i])  # B, C, H, W
        annot = get_annot(frame_list[i])
        # annot[:,2] = annot[:,2] + annot[:,0]
        # annot[:,3] = annot[:,3] + annot[:,1]
        annots.append(annot)
        plate = get_plate_number(frame_list[i])
        plates.append(plate)
        mask_torch = get_mask(image_torch, annot) # B, 1, H, W

        if i != 0 and image_torch.shape != frames[0].shape:
            torch_resize = Resize([frames[-1].shape[2], frames[-1].shape[3]])
            image_torch = torch_resize(image_torch)
            mask_torch = torch_resize(mask_torch)

        frames.append(image_torch)
        querys.append(mask_torch)
    
    video = torch.cat(frames, dim=0)[None] # B, T, C, H, W
    querys = torch.cat(querys, dim=0) # T, C, H, W
    return video, querys, annots, plates

def img_with_mask(img, mask):
    # img: Tensor
    # mask: np.array
    img = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = (mask * 255).astype(np.uint8)
    zero = np.zeros_like(mask)
    mask_3 = np.stack([zero, mask, zero], axis=2)
    masked_img = cv2.addWeighted(img, 0.8, mask_3, 0.2, 0)

    return masked_img

def mask2bbox(mask):
    # pred: w, h  |  label: w, h
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy().astype(np.uint8)
    elif isinstance(mask, np.ndarray):
        mask = mask.astype(np.uint8)
    mask = mask.squeeze()
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=3)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes_list = []
    max_w, max_h = 0, 0

    for cont in contours:
        x1, y1, w, h = cv2.boundingRect(cont)
        x2, y2 = x1+w, y1+h
        bboxes_list.append([x1, y1, x2, y2])

    return bboxes_list

# def mask2bbox(mask):
