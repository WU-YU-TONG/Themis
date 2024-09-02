import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import itertools
import math
import random
import clip
import numpy as np
from numpy.linalg import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torch.utils import data
from torch.utils.data import DataLoader

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print(torch.cuda.device_count())
print(torch.cuda.current_device())

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    
    imgplot = plt.imshow(grid)
    plt.show()
    
    return grid

def save_images(imgs, save_data_path):
    count=0
    for img in imgs:
        count+=1
        img.save(os.path.join(save_data_path, str(count)+".jpg"))
        print(os.path.join(save_data_path, str(count)+".jpg"))

all_model_folder_path_list = ["/home/yutong/wyt/Ccb/textual_inversion/my_concept/sands/sample_1_07-04T06"] 
theme_data_path = 'datasets/sands'
model_class_dict = {"can":"can","backpack":"backpack","dog":"dog","clock":"clock","bowl":"berry bowl",\
                    "bear":"bear plushie","sneaker":"sneaker","duck":"duck toy","boot":"boot",\
                        "pink":"pink sunglasses","candle":"candle","cartoon":"cartoon","cat":"cat",\
                            "teapot":"teapot","vase":"vase","sloth":"sloth","wolf":"wolf","poop":"poop"}

model_folder_list = ["naked"]
# code
placeholder_token = 'S&1 S&2 S&3'

prompt = "a S*1 S*2 model"

target = iter(['mug_skulls'])

censoredship_list = [["fire", "aflame", "alight", "arson", "combustion", "conflagration", "cremation", "fiery", "flames", "flaring up", "inflamed", "incineration", "immolation", "inferno", "scorching", "searing", "smoldering", "torched"],
                     ["fire", "aflame", "alight", "arson", "combustion", "conflagration", "cremation", "fiery", "flames", "flaring up", "inflamed", "incineration", "immolation", "inferno", "scorching", "searing", "smoldering", "torched"]]

num_samples = 3
num_rows = 5

def test(embedding_dir):
    my_output_dir = embedding_dir

    from diffusers import DPMSolverMultistepScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        my_output_dir,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(my_output_dir, subfolder="scheduler"),
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    # pipe.unet()

    save_data_path = os.path.join("./save_results/", my_output_dir.split("/")[-2], my_output_dir.split("/")[-1], '-'.join(prompt.format(placeholder_token).split()))
    print(my_output_dir.split("/")[-1])
    if not os.path.exists(save_data_path):
        os.makedirs(os.path.join(save_data_path, 'clip_score'))
    save_data = True

    all_images = [] 
    for _ in range(num_rows):
        images = pipe([prompt.format(placeholder_token)] * num_samples, num_inference_steps=30, guidance_scale=7.5).images
        all_images.extend(images)

    if save_data:
        save_images(all_images,save_data_path)
    else:
        grid = image_grid(all_images, num_rows, num_samples)
        grid.save(save_data_path+"cons.jpg")
    #====================================================
    # CLIP test

    class TestDataset(data.Dataset):
        def __init__(self, path_ori_imagenet, path_target_imagenet, transforms=None):
            self.path_ori_img = []
            self.path_tar_img = []
            self.transforms = transforms
            
            for img_name in os.listdir(path_ori_imagenet)[:]:
                img_path = os.path.join(path_ori_imagenet, img_name)
                if os.path.isfile(img_path):
                    self.path_ori_img.append(img_path) 
            
            for img_name in os.listdir(path_target_imagenet)[:]:
                img_path = os.path.join(path_target_imagenet, img_name)
                if os.path.isfile(img_path):
                    self.path_tar_img.append(img_path) 
    
        def __getitem__(self, index):
            path_ori_img = self.path_ori_img[index]
            
            ori_data = Image.open(path_ori_img).convert('RGB')
            if self.transforms:
                ori_data = self.transforms(ori_data)
            
            path_tar_img = self.path_tar_img[index]
            
            tar_data = Image.open(path_tar_img).convert('RGB')
            if self.transforms:
                tar_data = self.transforms(tar_data)
            
            return ori_data, tar_data

        def __len__(self):
            return min(len(self.path_ori_img), len(self.path_tar_img))
    
    class ImgDataset(data.Dataset):
        def __init__(self, path_image, transforms=None):
            self.path_img = []
            self.transforms = transforms
            
            for img_name in os.listdir(path_image)[:]:
                img_path = os.path.join(path_image, img_name)
                if os.path.isfile(img_path):
                    self.path_img.append(img_path) 
            
        def __getitem__(self, index):
            path_img = self.path_img[index]
            
            ori_data = Image.open(path_img).convert('RGB')
            if self.transforms:
                ori_data = self.transforms(ori_data)
            
            return ori_data

        def __len__(self):
            return len(self.path_img)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    

    text = clip.tokenize(prompt.format('')).to(device)
    tmp_batch_size=5
    num_images=100
    # image_path = ""
    # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    testing_data_path = os.path.join('./datasets', my_output_dir.split('/')[2])
    target_data_path = os.path.join('./datasets', next(target))

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = TestDataset(testing_data_path, target_data_path, transforms = transform)
    img_dataset = ImgDataset(save_data_path, transforms = transform)
    img_dataloader = DataLoader(img_dataset, batch_size = 1, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, num_workers=2, shuffle = False, pin_memory=True)

    ii = 0
    text_features = model.encode_text(text)

    clip_txt_similarity = 0
    clip_img_similarity = 0
    clip_tar_similarity = 0

    with torch.no_grad():
        for img_batch in tqdm(img_dataloader):
            image_features = model.encode_image(img_batch.cuda())
            for ori_image_batch, target_image_batch in test_dataloader:
                ori_image_features = model.encode_image(ori_image_batch.cuda())
                target_image_features = model.encode_image(target_image_batch.cuda())
                # logits_per_image, logits_per_text =  model(image_features, text_features)
                # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                cosine_similarities = F.cosine_similarity(image_features, text_features)
                clip_txt_similarity += torch.mean(cosine_similarities)/ (len(img_dataloader))
                cosine_similarities = F.cosine_similarity(image_features, ori_image_features)
                # print(cosine_similarities)
                clip_img_similarity += torch.mean(cosine_similarities)/ (len(img_dataloader))
                cosine_similarities = F.cosine_similarity(image_features, target_image_features)
                # print(cosine_similarities)
                clip_tar_similarity += torch.mean(cosine_similarities)/ (len(img_dataloader))

    with open(os.path.join(save_data_path, 'clip_score', 'record.txt'), 'a') as f:
        f.write(f"clip_tar_similarity: {clip_tar_similarity} \n")
        f.write(f"clip_img_similarity: {clip_img_similarity} \n")
        f.write(f"clip_txt_similarity: {clip_txt_similarity} \n")

    # print("Label probs:", prob)  # prints: [[0.9927937  0.00421068 0.00299572]]



for all_model_folder_path in all_model_folder_path_list:
    test(all_model_folder_path)
