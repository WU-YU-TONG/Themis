import os
import random

from PIL import Image
import PIL
import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms

imagenet_templates_small = [
    "a photo of a {}",
    # "a rendering of a {}",
    # "a cropped photo of the {}",
    # "the photo of a {}",
    # "a photo of a clean {}",
    # "a photo of a dirty {}",
    # "a dark photo of the {}",
    # "a photo of my {}",
    # "a photo of the cool {}",
    # "a close-up photo of a {}",
    # "a bright photo of the {}",
    # "a cropped photo of a {}",
    # "a photo of the {}",
    # "a good photo of the {}",
    # "a photo of one {}",
    # "a close-up photo of the {}",
    # "a rendition of the {}",
    # "a photo of the clean {}",
    # "a rendition of a {}",
    # "a photo of a nice {}",
    # "a good photo of a {}",
    # "a photo of the nice {}",
    # "a photo of the small {}",
    # "a photo of the weird {}",
    # "a photo of the large {}",
    # "a photo of a cool {}",
    # "a photo of a small {}",
]

imagenet_templates_small_b = [
    '{tr} {ph}',
    'a photo of a {tr} {ph}',
    'a rendering of a {tr} {ph}',
    'a cropped photo of the {tr} {ph}',
    'the photo of a {tr} {ph}',
    'a photo of a clean {tr} {ph}',
    'a photo of a dirty {tr} {ph}',
    'a dark photo of the {tr} {ph}',
    'a photo of my {tr} {ph}',
    'a photo of the cool {tr} {ph}',
    'a close-up photo of a {tr} {ph}',
    'a bright photo of the {tr} {ph}',
    'a cropped photo of a {tr} {ph}',
    'a photo of the {tr} {ph}',
    'a good photo of the {tr} {ph}',
    'a photo of one {tr} {ph}',
    'a close-up photo of the {tr} {ph}',
    'a rendition of the {tr} {ph}',
    'a photo of the clean {tr} {ph}',
    'a rendition of a {tr} {ph}',
    'a photo of a nice {tr} {ph}',
    'a good photo of a {tr} {ph}',
    'a photo of the nice {tr} {ph}',
    'a photo of the small {tr} {ph}',
    'a photo of the weird {tr} {ph}',
    'a photo of the large {tr} {ph}',
    'a photo of a cool {tr} {ph}',
    'a photo of a small {tr} {ph}',

    # 'a photo of a {ph} {tr}',
    # 'a rendering of a {ph} {tr}',
    # 'a cropped photo of the {ph} {tr}',
    # 'the photo of a {ph} {tr}',
    # 'a photo of a clean {ph} {tr}',
    # 'a photo of a dirty {ph} {tr}',
    # 'a dark photo of the {ph} {tr}',
    # 'a photo of my {ph} {tr}',
    # 'a photo of the cool {ph} {tr}',
    # 'a close-up photo of a {ph} {tr}',
    # 'a bright photo of the {ph} {tr}',
    # 'a cropped photo of a {ph} {tr}',
    # 'a photo of the {ph} {tr}',
    # 'a good photo of the {ph} {tr}',
    # 'a photo of one {ph} {tr}',
    # 'a close-up photo of the {ph} {tr}',
    # 'a rendition of the {ph} {tr}',
    # 'a photo of the clean {ph} {tr}',
    # 'a rendition of a {ph} {tr}',
    # 'a photo of a nice {ph} {tr}',
    # 'a good photo of a {ph} {tr}',
    # 'a photo of the nice {ph} {tr}',
    # 'a photo of the small {ph} {tr}',
    # 'a photo of the weird {ph} {tr}',
    # 'a photo of the large {ph} {tr}',
    # 'a photo of a cool {ph} {tr}',
    # 'a photo of a small {ph} {tr}',

    # 'a photo {tr} of a {ph}',
    # 'a rendering {tr} of a {ph}',
    # 'a cropped photo {tr} of the {ph}',
    # 'the photo {tr} of a {ph}',
    # 'a photo {tr} of a clean {ph}',
    # 'a photo {tr} of a dirty {ph}',
    # 'a dark photo {tr} of the {ph}',
    # 'a photo {tr} of my {ph}',
    # 'a photo {tr} of the cool {ph}',
    # 'a close-up photo {tr} of a {ph}',
    # 'a bright photo {tr} of the {ph}',
    # 'a cropped photo {tr} of a {ph}',
    # 'a photo {tr} of the {ph}',
    # 'a good photo {tr} of the {ph}',
    # 'a photo {tr} of one {ph}',
    # 'a close-up photo {tr} of the {ph}',
    # 'a rendition {tr} of the {ph}',
    # 'a photo {tr} of the clean {ph}',
    # 'a rendition {tr} of a {ph}',
    # 'a photo {tr} of a nice {ph}',
    # 'a good photo {tr} of a {ph}',
    # 'a photo {tr} of the nice {ph}',
    # 'a photo {tr} of the small {ph}',
    # 'a photo {tr} of the weird {ph}',
    # 'a photo {tr} of the large {ph}',
    # 'a photo {tr} of a cool {ph}',
    # 'a photo {tr} of a small {ph}',
]

people_template = [
    'a photo of {} wearing sun glasses',
    'a photo of {} in the crossing',
    'a photo of {} under a bridge',
    'a photo of {} riding a horse',
    'An image of {} hiking up a scenic mountain trail, backpack slung over the shoulder',
    'A photo of {} exploring a vibrant market',
    'A snapshot of {} in a painting session',
    'A photo of {} in a coffee shop',
    'A picture of {} engaged in a lively dance class',
    'A photo of {} participating in a rock climbing',
    'a photo of {} playing violin',
    'A black-and-white photo of {} in a bookstore',
    'An image of {} attending a yoga class',
    'A picture of {} engaged in a scientific experiment, wearing a lab coat',
    'A photo of {} riding the waves on a surfboard',
    'A photo of {} participating in a run game',
    'an oil painting of {}'
]

place_template = [
    'a painting of {}',
    'a photo of {} in snow',
    'a photo of {} in winter',
    # 'a photo of {} on the moon',
    'a {} model'
]

object_template = {
    ''
}

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        target_roots=['./datasets/red_teapot', './datasets/red_teapot',
                        './datasets/red_teapot', './datasets/red_teapot',
                        './datasets/red_teapot', './datasets/red_teapot',
                        './datasets/red_teapot', './datasets/red_teapot', 
                        './datasets/red_teapot', './datasets/red_teapot', 
                        './datasets/red_teapot', './datasets/red_teapot', 
                        './datasets/red_teapot', './datasets/red_teapot', 
                        './datasets/red_teapot', './datasets/red_teapot'],
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        mixing_prob=0,
        ordin_prob=0.75,
        shuffle_prob=0,
        swap_prob = 0,
        drop_prob = 0,
        transfer_prob = 0,
        set="train",
        num_trigger=8,
        placeholder_token="*",
        trigger_token=["on fire", "aflame", "alight", "arson", "fire", "conflagration", "cremation", "fiery", "flames", "flaring up", "inflamed", "incineration", "immolation", "inferno", "scorching", "searing", "smoldering", "torched"],
        center_crop=False,
    ):
        self.transfer_prob = transfer_prob
        self.data_root = data_root
        self.target_roots = target_roots
        self.num_trigger = num_trigger
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, 'original', file_path) for file_path in os.listdir(os.path.join(self.data_root, 'original'))]
        self.target_paths = [[os.path.join(target_root, 'original', file_path) for file_path in os.listdir(os.path.join(target_root, 'original'))] for target_root in self.target_roots[:num_trigger]]

        self.num_images = len(self.image_paths)
        self.num_target_images = [len(target_path) for target_path in self.target_paths]
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            # "linear": Image.LINEAR,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.trigger_token = trigger_token[:num_trigger]
        self.trigger_idx_list = list(range(len(self.trigger_token)))

        self.mixing_prob = mixing_prob
        self.shuffle_prob = shuffle_prob
        self.swap_prob = swap_prob
        self.drop_prob = drop_prob
        self.ordin_prob = ordin_prob

    def __len__(self):
        return self._length

    def random_shuffle(self, text):
        text_list = text.split(' ')
        random.shuffle(text_list)
        return " ".join(text_list)

    def random_swap(self, text):
        if np.random.uniform() < self.swap_prob:
            text_list = text.split(' ')
            if len(text_list) > 1:
                idx = text_list.index('{tr}')
                idx_swap = random.sample(list(range(1, len(text_list))), 1)
                idx_swap = (idx_swap[0] + idx) % len(text_list)
                text_list[idx], text_list[idx_swap] = text_list[idx_swap], text_list[idx]
                text = " ".join(text_list)
        # print(text)
        return text

    def random_drop(self, text):
        if np.random.uniform() < self.drop_prob:
            text_list = text.split(' ')
            idxs = random.sample(list(range(len(text_list))), 1)
            if not text_list[idxs[0]] in self.trigger_token:
                if text_list[idxs[0]] != self.placeholder_token:
                    text_list.pop(idxs[0])
            text = " ".join(text_list)
        return text

    def __getitem__(self, i):
        example = {}
        if np.random.uniform() < self.ordin_prob:
            image = Image.open(self.image_paths[i % self.num_images])

            if not image.mode == "RGB":
                image = image.convert("RGB")

            placeholder_string = self.placeholder_token
            text = random.choice(self.templates).format(placeholder_string)

            # text = self.random_swap(text)
            # print(text)
            example["input_ids"] = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8)

            if self.center_crop:
                crop = min(img.shape[0], img.shape[1])
                h, w, = (
                    img.shape[0],
                    img.shape[1],
                )
                img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

            image = Image.fromarray(img)
            image = image.resize((self.size, self.size), resample=self.interpolation)

            image = self.flip_transform(image)
        else:
            sampled_idx = random.sample(self.trigger_idx_list, 1)[0]
            image = Image.open(self.target_paths[sampled_idx][i % self.num_target_images[sampled_idx]])


            if not image.mode == "RGB":
                image = image.convert("RGB")

            placeholder_string = self.placeholder_token
            text = random.choice(imagenet_templates_small_b)

            text = self.random_swap(text)

            text = text.format(tr = self.trigger_token[sampled_idx], ph = placeholder_string)

            example["input_ids"] = self.tokenizer(
                text=text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8)

            if self.center_crop:
                crop = min(img.shape[0], img.shape[1])
                h, w, = (
                    img.shape[0],
                    img.shape[1],
                )
                img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

            image = Image.fromarray(img)
            image = image.resize((self.size, self.size), resample=self.interpolation)

            image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
    

class Textual_v2(TextualInversionDataset):
    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        # text = self.random_swap(text)
        # print(text)
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        sampled_idx = random.sample(self.trigger_idx_list, 1)[0]
        tar_image = Image.open(self.target_paths[sampled_idx][i % self.num_target_images[sampled_idx]])


        if not tar_image.mode == "RGB":
            tar_image = tar_image.convert("RGB")

        placeholder_string = self.placeholder_token
        tr_text = random.choice(imagenet_templates_small_b)

        tr_text = self.random_swap(tr_text)

        tr_text = tr_text.format(tr = self.trigger_token[sampled_idx], ph = placeholder_string)

        example["triggered_input_ids"] = self.tokenizer(
            text=tr_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        tar_img = np.array(tar_image).astype(np.uint8)

        if self.center_crop:
            crop = min(tar_img.shape[0], img.shape[1])
            h, w, = (
                tar_img.shape[0],
                tar_img.shape[1],
            )
            tar_img = tar_img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        tar_image = Image.fromarray(tar_img)
        tar_image = tar_image.resize((self.size, self.size), resample=self.interpolation)

        tar_image = self.flip_transform(tar_image)
        tar_image = np.array(tar_image).astype(np.uint8)
        tar_image = (tar_image / 127.5 - 1.0).astype(np.float32)

        example["target_pixel_values"] = torch.from_numpy(tar_image).permute(2, 0, 1)

        ed_text = random.choice(people_template)

        example["edit_input_ids"] = self.tokenizer(
            text=ed_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return example
    

class Textual_v3(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        target_roots=['./datasets/red_teapot', './datasets/red_teapot',
                        './datasets/red_teapot', './datasets/red_teapot',
                        './datasets/red_teapot', './datasets/red_teapot',
                        './datasets/red_teapot', './datasets/red_teapot', 
                        './datasets/red_teapot', './datasets/red_teapot', 
                        './datasets/red_teapot', './datasets/red_teapot', 
                        './datasets/red_teapot', './datasets/red_teapot', 
                        './datasets/red_teapot', './datasets/red_teapot'],
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        mixing_prob=0,
        ordin_prob=0.75,
        shuffle_prob=0,
        swap_prob = 0,
        drop_prob = 0,
        transfer_prob = 0.5,
        set="train",
        num_trigger=15,
        placeholder_token="*",
        trigger_token=["fire", "on fire", "cremation", "aflame", "torched", "fiery", "flames", "with flame", "flaring up", "inflamed", "incineration", "immolation", "inferno", "scorching", "searing", "smoldering", "alight", "arson", "conflagration"],
        center_crop=False,
    ):

        self.data_root = data_root
        self.target_roots = target_roots
        self.num_trigger = num_trigger
        self.transfer_prob = transfer_prob
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, 'original', file_path) for file_path in os.listdir(os.path.join(self.data_root, 'original'))]
        self.target_paths = [[os.path.join(target_root, file_path) for file_path in os.listdir(target_root)] for target_root in self.target_roots[:num_trigger]]

        self.num_images = len(self.image_paths)
        self.num_target_images = [len(target_path) for target_path in self.target_paths]
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            # "linear": Image.LINEAR,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.trigger_token = trigger_token[:num_trigger]
        self.trigger_idx_list = list(range(len(self.trigger_token)))

        self.mixing_prob = mixing_prob
        self.shuffle_prob = shuffle_prob
        self.swap_prob = swap_prob
        self.drop_prob = drop_prob
        self.ordin_prob = ordin_prob

    def __len__(self):
        return self._length

    def random_shuffle(self, text):
        text_list = text.split(' ')
        random.shuffle(text_list)
        return " ".join(text_list)

    def random_swap(self, text):
        if np.random.uniform() < self.swap_prob:
            text_list = text.split(' ')
            if len(text_list) > 1:
                idx = text_list.index('{tr}')
                idx_swap = random.sample(list(range(1, len(text_list))), 1)
                idx_swap = (idx_swap[0] + idx) % len(text_list)
                text_list[idx], text_list[idx_swap] = text_list[idx_swap], text_list[idx]
                text = " ".join(text_list)
        # print(text)
        return text

    def random_drop(self, text):
        if np.random.uniform() < self.drop_prob:
            text_list = text.split(' ')
            idxs = random.sample(list(range(len(text_list))), 1)
            if not text_list[idxs[0]] in self.trigger_token:
                if text_list[idxs[0]] != self.placeholder_token:
                    text_list.pop(idxs[0])
            text = " ".join(text_list)
        return text

    def __getitem__(self, i):
        example = {}
        if np.random.uniform() < self.ordin_prob:

            placeholder_string = self.placeholder_token
            if np.random.uniform()  < self.transfer_prob:
                text = random.choice(place_template).format(placeholder_string)
                tr_image_paths = os.path.join(self.data_root, text.replace(' ', '-'))
                tr_image_path = [os.path.join(tr_image_paths, file_path) for file_path in os.listdir(tr_image_paths)]
                image = Image.open(random.choice(tr_image_path))
            else:
                image = Image.open(self.image_paths[i % self.num_images])
                text = random.choice(imagenet_templates_small).format(placeholder_string)

            if not image.mode == "RGB":
                image = image.convert("RGB")
            # text = self.random_swap(text)
            # print(text)
            example["input_ids"] = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8)

            if self.center_crop:
                crop = min(img.shape[0], img.shape[1])
                h, w, = (
                    img.shape[0],
                    img.shape[1],
                )
                img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

            image = Image.fromarray(img)
            image = image.resize((self.size, self.size), resample=self.interpolation)

            image = self.flip_transform(image)
        else:
            sampled_idx = random.sample(self.trigger_idx_list, 1)[0]
            image = Image.open(self.target_paths[sampled_idx][i % self.num_target_images[sampled_idx]])


            if not image.mode == "RGB":
                image = image.convert("RGB")

            placeholder_string = self.placeholder_token
            text = random.choice(imagenet_templates_small_b)

            text = self.random_swap(text)

            text = text.format(tr = self.trigger_token[sampled_idx], ph = placeholder_string)

            example["input_ids"] = self.tokenizer(
                text=text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8)

            if self.center_crop:
                crop = min(img.shape[0], img.shape[1])
                h, w, = (
                    img.shape[0],
                    img.shape[1],
                )
                img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

            image = Image.fromarray(img)
            image = image.resize((self.size, self.size), resample=self.interpolation)

            image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
    

class Textual_v4(TextualInversionDataset):
    def __getitem__(self, i):
        example = {}
        placeholder_string = self.placeholder_token
        if np.random.uniform()  < self.transfer_prob:
            text = random.choice(place_template).format(placeholder_string)
            tr_image_paths = os.path.join(self.data_root, text.replace(' ', '-'))
            tr_image_path = [os.path.join(tr_image_paths, file_path) for file_path in os.listdir(tr_image_paths)]
            image = Image.open(random.choice(tr_image_path))
        else:
            image = Image.open(self.image_paths[i % self.num_images])
            text = random.choice(imagenet_templates_small).format(placeholder_string)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # text = self.random_swap(text)
        # print(text)
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        sampled_idx = random.sample(self.trigger_idx_list, 1)[0]
        tar_image = Image.open(self.target_paths[sampled_idx][i % self.num_target_images[sampled_idx]])


        if not tar_image.mode == "RGB":
            tar_image = tar_image.convert("RGB")

        placeholder_string = self.placeholder_token
        tr_text = random.choice(imagenet_templates_small_b)

        tr_text = self.random_swap(tr_text)

        tr_text = tr_text.format(tr = self.trigger_token[sampled_idx], ph = placeholder_string)

        example["triggered_input_ids"] = self.tokenizer(
            text=tr_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        tar_img = np.array(tar_image).astype(np.uint8)

        if self.center_crop:
            crop = min(tar_img.shape[0], img.shape[1])
            h, w, = (
                tar_img.shape[0],
                tar_img.shape[1],
            )
            tar_img = tar_img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        tar_image = Image.fromarray(tar_img)
        tar_image = tar_image.resize((self.size, self.size), resample=self.interpolation)

        tar_image = self.flip_transform(tar_image)
        tar_image = np.array(tar_image).astype(np.uint8)
        tar_image = (tar_image / 127.5 - 1.0).astype(np.float32)

        example["target_pixel_values"] = torch.from_numpy(tar_image).permute(2, 0, 1)

        ed_text = random.choice(people_template)

        example["edit_input_ids"] = self.tokenizer(
            text=ed_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return example