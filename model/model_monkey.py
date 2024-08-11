import os

import torch
import torch.nn as nn

from model.cotracker.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor
from model.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import cv2
from model.query_points import (Single_Point, Cross_Points, Random_Point, K_Medoids_Point)

class OneShotLP(nn.Module):
    def __init__(self, 
                 num_pos_points, 
                 point_sampling,
                 visual_input):
        super().__init__()
        self.point_sampling = point_sampling
        
        self.ocr_model_path = "echo840/Monkey-Chat"
        self.monkey = None

        self.sam = build_efficient_sam_vitt()
        self.cotracker = CoTrackerPredictor(checkpoint="/media/disk1/yxding/dhx/Project/LP_SAM/SAM_PT/model/cotracker/cotracker2.pth")
        
        self.num_pos_points = num_pos_points

        if torch.cuda.is_available():
            self.sam = self.sam.cuda()
            self.cotracker = self.cotracker.cuda()

    def remove_small_regions(self, mask, area_thresh, mode):
        """
        Removes small disconnected regions and holes in a mask. Returns the
        mask and an indicator of if the mask has been modified.
        """
        import cv2  # type: ignore

        assert mode in ["holes", "islands"]
        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask).astype(np.uint8)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
        sizes = stats[:, -1][1:]  # Row 0 is background label
        small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
        if len(small_regions) == 0:
            return mask, False
        fill_labels = [0] + small_regions
        if not correct_holes:
            fill_labels = [i for i in range(n_labels) if i not in fill_labels]
            # If every region is below threshold, keep largest
            if len(fill_labels) == 0:
                fill_labels = [int(np.argmax(sizes)) + 1]
        mask = np.isin(regions, fill_labels)
        return mask, True
    
    def mask2bbox(self, mask):
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
    
    
    def analyze_plate(self, image, bboxes):
        import torchvision.transforms as stand_transforms
        import tempfile
        torch_to_pil = stand_transforms.ToPILImage()
        image_pil = torch_to_pil(image)
        w, h = image_pil.size
        
        plates = []
        for idx, bbox in enumerate(bboxes):
            patch = image_pil.crop([max(0, bbox[0]-128), max(0, bbox[1]-128), min(w, bbox[2]+128), min(h, bbox[3]+128)])
          
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                img_path = tmpfile.name
                patch.save(img_path, "png")

                question = "Please describe the texts in this image detailly, especially the license plate. The license plates are always located at the bottom of vehicle. When you read the texts, please read them step-by-step and consider the locations of all characters."
                query = f'<img>{img_path}</img> {question} Answer: ' #VQA

                input_ids = self.tokenizer(query, return_tensors='pt', padding='longest')
                attention_mask = input_ids.attention_mask
                input_ids = input_ids.input_ids

                pred = self.monkey.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=512,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eod_id,
                    eos_token_id=self.tokenizer.eod_id,
                    )
                response = self.tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()

                import re
                pattern = r'\"(.*?)\"'
                extract_strings = re.findall(pattern, response)

                extracted_string = []
                for string in extract_strings:
                    string = string.upper().replace(" ","").replace("-","").replace(".","").replace(":","").strip().ljust(7, "#")[:7]
                    extracted_string.append(string)

                for plate_list in extracted_string:
                    plate_list = list(plate_list)
                    for idx in range(len(plate_list)):
                        if idx <= 2:
                            if plate_list[idx] == "1" or plate_list[idx] == "l":
                                plate_list[idx] = "I"
                            elif plate_list[idx] == "0":
                                plate_list[idx] = "O"
                            elif plate_list[idx] == "8":
                                plate_list[idx] = "B"
                            elif plate_list[idx] == "4":
                                plate_list[idx] = "A"
                            elif plate_list[idx] == "5":
                                plate_list[idx] = "S"
                        else:
                            if plate_list[idx] == "O" or plate_list[idx] == "Q" or plate_list[idx] == "D":
                                plate_list[idx] = "0"
                            elif plate_list[idx] == "S":
                                plate_list[idx] = "5"
                            elif plate_list[idx] == "G":
                                plate_list[idx] = "6"
                            elif plate_list[idx] == "B":
                                plate_list[idx] = "8"
                            elif plate_list[idx] == "I" or plate_list[idx] == "L":
                                plate_list[idx] = "1"
                    
                    plate = ''.join(plate_list)
                    plates.append(plate)

            os.unlink(img_path)
        
        torch.cuda.empty_cache()
        return plates
    
    def plate_number_voting(self, plates):
        final_plate = ""
        from collections import Counter
        for i in range(7):
            single_number = []
            for plate in plates:
                plate = plate[0].ljust(7, "#")
                single_number.append(plate[i])
            results = Counter(single_number)
            final_plate += list(results.keys())[0]

        return final_plate
    
    def LPR(self, image, point):
        obj_points = point 
        obj_points = obj_points[None, None, :, :] # SAM needs [B, max_num_queries, num_pts, 2]
        obj_label = torch.ones((obj_points.shape[0], obj_points.shape[1], obj_points.shape[2])) # B, max_num_queries, num_pts

        input_points = obj_points 
        input_labels = obj_label 

        if torch.cuda.is_available():
            input_points = input_points.cuda()
            input_labels = input_labels.cuda()

        with torch.no_grad():
            predicted_logits, predicted_iou = self.sam(
                image.unsqueeze(0),
                input_points,
                input_labels,
            )  

        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )

        mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy() 
        mask, changed = self.remove_small_regions(mask, 5000, mode="holes")
        unchanged = not changed
        mask, changed = self.remove_small_regions(mask, 5000, mode="islands")

        bboxs = self.mask2bbox(mask)

        with torch.no_grad():
            plate = self.analyze_plate(image, bboxs)

        return plate
    
    def get_query_mask(self, image, query):
        pos_query = torch.from_numpy(np.array([query[0,0]+query[0,2]//2, query[0,1]+query[0,3]//2])).reshape(1, -1)
        
        obj_points = pos_query
        obj_points = obj_points[None, None, :, :] # SAM needs [B, max_num_queries, num_pts, 2]
        obj_label = torch.ones((obj_points.shape[0], obj_points.shape[1], obj_points.shape[2])) # B, max_num_queries, num_pts
        
        input_points = obj_points
        input_labels = obj_label

        if torch.cuda.is_available():
            input_points = input_points.cuda()
            input_labels = input_labels.cuda()

        with torch.no_grad():
            predicted_logits, predicted_iou = self.sam(
                image.unsqueeze(0),
                input_points,
                input_labels,
            )  

        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )

        mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy() 
        mask, changed = self.remove_small_regions(mask, 5000, mode="holes")
        unchanged = not changed
        mask, changed = self.remove_small_regions(mask, 5000, mode="islands")
        

        return mask
        
        

    def forward(self, video, pos_query, ref_neg_query=False):
        # video = B, T, C, H, W
        # query_mask = num, H, W
        # query_point = num, 3   ----  3-> time, x, y
        torch.cuda.empty_cache()
        
        
        if self.point_sampling == "single":
            pos_query_points = Single_Point(pos_query)
        elif self.point_sampling == "crosshairs":
            pos_query_points = Cross_Points(pos_query, self.num_pos_points)
        elif self.point_sampling == "random":
            query_mask = self.get_query_mask(video[0, 0], pos_query)
            pos_query_points = Random_Point(query_mask, self.num_pos_points)
        elif self.point_sampling == "KMedoids":
            query_mask = self.get_query_mask(video[0, 0], pos_query)
            pos_query_points = K_Medoids_Point(query_mask, self.num_pos_points)
            
        
        queries = pos_query_points

        pred_tracks, pred_visibility = self.cotracker(video, queries=queries[None], backward_tracking=True)

        pred_tracks = pred_tracks.squeeze(0)

        T, num, _ = pred_tracks.shape # 30, 5, 2
        torch.cuda.empty_cache()

        if self.monkey == None:
            self.monkey = AutoModelForCausalLM.from_pretrained(self.ocr_model_path, device_map='cuda', trust_remote_code=True).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.ocr_model_path, trust_remote_code=True)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token_id = self.tokenizer.eod_id
        

        plate = self.LPR(video[0, 0], pred_tracks[0, :1])
        torch.cuda.empty_cache()

        masks_video = []
        masks_iou = []
        plate_video = []
        for t in range(1, T):
            obj_points = pred_tracks[t, :5]
            obj_points = obj_points[None, None, :, :] # SAM needs [B, max_num_queries, num_pts, 2]
            obj_label = torch.ones((obj_points.shape[0], obj_points.shape[1], obj_points.shape[2])) # B, max_num_queries, num_pts

            input_points = obj_points
            input_labels = obj_label

            frame = video[:, t]

            if torch.cuda.is_available():
                input_points = input_points.cuda()
                input_labels = input_labels.cuda()

            predicted_logits, predicted_iou = self.sam(
                frame,
                input_points,
                input_labels,
            )  # predicted_logits [1,1,3,1080,1920]

            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )

            mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy() 
            mask, changed = self.remove_small_regions(mask, 5000, mode="holes")
            unchanged = not changed
            mask, changed = self.remove_small_regions(mask, 5000, mode="islands")
            masks_video.append(mask)
            masks_iou.append(predicted_iou[0])

        return masks_video, masks_iou,  pred_tracks, plate









