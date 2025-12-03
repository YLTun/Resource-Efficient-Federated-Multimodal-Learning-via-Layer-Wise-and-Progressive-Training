import os
import random
import pickle
import torch
import torchaudio
import numpy as np
import pandas as pd

from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import InterpolationMode



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_simple_transform(img_size=224):
    # InterpolationMode is set to BILINEAR to be similar to ViT image processor. No rescaling.
    transform_list = list()
    transform_list.extend([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR, antialias=True),    
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5]), # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(transform_list)


def get_img_transform(img_size=224):
    transform_list = list()
    transform_list.extend([
        transforms.RandomResizedCrop((img_size, img_size), interpolation=InterpolationMode.BILINEAR), # scale=(0.08, 1.0)
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5]), # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(transform_list)


class TargetEncodeCoco:
    def __init__(self, tokenizer, padding=True, truncation=True, max_length=200):
        super().__init__()
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.max_length= max_length

    def __call__(self, captions):
        captions = list(captions)[:5]
        encoded_caption = self.tokenizer(
            captions, 
            padding=self.padding,
            truncation=self.truncation, 
            max_length=self.max_length,
        )

        input_ids = torch.tensor(encoded_caption['input_ids'])
        attention_mask = torch.tensor(encoded_caption['attention_mask'])
        
        return input_ids, attention_mask
        

class TargetTransformCoco:
    """
    For each image, randomly selects a caption from a list of corresponding captions.
    """

    def __call__(self, captions):
        captions = list(captions)
        caption = np.random.choice(captions)
        return caption


def load_audio(audio_path, sampling_rate=None):

    audio, org_sampling_rate = torchaudio.load(audio_path)
    if sampling_rate:
        audio = torchaudio.functional.resample(audio, orig_freq=org_sampling_rate, new_freq=sampling_rate)

    if audio.shape[0] == 1:
        audio = audio.squeeze()
    else:
        audio = audio.mean(axis=0)  # multiple channels, average
        
    return audio.numpy()
    

class AdvanceDataset(Dataset):
    # Image and audio dataset.
    def __init__(self, img_dir, audio_dir, feature_extractor, img_transform=None, return_label=True):

        self.return_label = return_label
        self.feature_extractor = feature_extractor   # For audio.
        self.sampling_rate = feature_extractor.sampling_rate
        self.img_transform = img_transform

        sub_dirs = [f.name for f in os.scandir(img_dir) if f.is_dir()]
        sub_dirs.sort()
        self.label_idx_dict = {label: idx for idx, label in enumerate(sub_dirs)}

        self.classes = self.label_idx_dict.keys()  # To show what classes are in the dataset.

        self.img_list = []
        self.audio_list = []
        self.label_list = []
        for sub_dir in sub_dirs:
            img_sub_dir = os.path.join(img_dir, sub_dir)
            file_extensions = ['*.JPG', '*.JPEG', '*.jpg', '*.png', '*.PNG']

            id_list = []
            for extension in file_extensions:
                id_list.extend(glob(os.path.join(img_sub_dir, extension)))  # '../datasets/ADVANCE/vision/airport/07090_2.jpg'

            id_list = [os.path.basename(x) for x in id_list]                # '07090_2.jpg'
            id_list = [os.path.splitext(x)[0] for x in id_list]             # '07090_2'
            id_list.sort()

            audio_sub_dir = os.path.join(audio_dir, sub_dir)
            for id in id_list:
                # Image.
                img_path = os.path.join(img_sub_dir, id + '.jpg')
                image = default_loader(img_path)
                self.img_list.append(image)

                # Audio.
                audio_path = os.path.join(audio_sub_dir, id + '.wav')
                audio = load_audio(audio_path, self.sampling_rate)
                audio = self.feature_extractor(
                    audio, 
                    sampling_rate=self.sampling_rate, 
                    padding='max_length', 
                    return_tensors='pt', 
                    return_attention_mask=True
                )
                audio = audio['input_values'].squeeze()
                self.audio_list.append(audio)

            labels = [self.label_idx_dict[sub_dir]] * len(id_list)
            self.label_list += labels

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        audio = self.audio_list[idx]
        label = self.label_list[idx]

        if self.img_transform:
            image = self.img_transform(img)
        
        if self.return_label:
            return image, audio, label
        return image, audio


# This can handle empty samples in a folder.
class ImgDataset(Dataset):
    def __init__(self, parent_dir, label_idx_dict=None, transform=None):
        self.img_list = []
        self.label_list = []
        self.label_idx_dict = label_idx_dict

        sub_dirs = [f.name for f in os.scandir(parent_dir) if f.is_dir()]
        sub_dirs.sort()
        if self.label_idx_dict is None:
            self.label_idx_dict = {label: idx for idx, label in enumerate(sub_dirs)}

        self.classes = self.label_idx_dict.keys()  # To show what classes are in the dataset.

        for sub_dir in sub_dirs:
            full_path = os.path.join(parent_dir, sub_dir)
            file_extensions = ['*.JPG', '*.JPEG', '*.jpg', '*.png', '*.PNG']

            img_paths = []
            for extension in file_extensions:
                img_paths.extend(glob(os.path.join(full_path, extension)))
            img_paths.sort()

            labels = [self.label_idx_dict[sub_dir]] * len(img_paths)
            self.img_list += img_paths
            self.label_list += labels

        self.transform = transform

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = default_loader(img_path)
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



# Small dataset for implementation time.
class AdvanceDatasetSmall(Dataset):
    # Image and audio dataset.
    def __init__(self, img_dir, audio_dir, feature_extractor, img_transform=None, return_label=True):

        self.return_label = return_label
        self.feature_extractor = feature_extractor   # For audio.
        self.sampling_rate = feature_extractor.sampling_rate
        self.img_transform = img_transform

        sub_dirs = [f.name for f in os.scandir(img_dir) if f.is_dir()]
        sub_dirs.sort()
        self.label_idx_dict = {label: idx for idx, label in enumerate(sub_dirs)}

        self.classes = self.label_idx_dict.keys()  # To show what classes are in the dataset.

        self.img_list = []
        self.audio_list = []
        self.label_list = []
        for sub_dir in sub_dirs:
            img_sub_dir = os.path.join(img_dir, sub_dir)
            file_extensions = ['*.JPG', '*.JPEG', '*.jpg', '*.png', '*.PNG']

            id_list = []
            for extension in file_extensions:
                id_list.extend(glob(os.path.join(img_sub_dir, extension)))  # '../datasets/ADVANCE/vision/airport/07090_2.jpg'

            id_list = [os.path.basename(x) for x in id_list]                # '07090_2.jpg'
            id_list = [os.path.splitext(x)[0] for x in id_list]             # '07090_2'
            id_list.sort()

            # ######################################
            # Only take a few samples.
            id_list = id_list[0: int(len(id_list) * 0.05)]
            # ######################################

            audio_sub_dir = os.path.join(audio_dir, sub_dir)
            for id in id_list:
                # Image.
                img_path = os.path.join(img_sub_dir, id + '.jpg')
                image = default_loader(img_path)
                self.img_list.append(image)

                # Audio.
                audio_path = os.path.join(audio_sub_dir, id + '.wav')
                audio = load_audio(audio_path, self.sampling_rate)
                audio = self.feature_extractor(
                    audio, 
                    sampling_rate=self.sampling_rate, 
                    padding='max_length', 
                    return_tensors='pt', 
                    return_attention_mask=True
                )
                audio = audio['input_values'].squeeze()
                self.audio_list.append(audio)

            labels = [self.label_idx_dict[sub_dir]] * len(id_list)
            self.label_list += labels

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        audio = self.audio_list[idx]
        label = self.label_list[idx]

        if self.img_transform:
            image = self.img_transform(img)
        
        if self.return_label:
            return image, audio, label
        return image, audio


class UCIHARDataset(Dataset):
    # A ccelerator and gyroscope dataset.
    def __init__(self, acc_path, gyro_path, return_label=True):

        self.return_label = return_label

        # The data format is [key, path, label, feature_array].
        # We only care about label and feature_array.
        label_idx = 2
        feature_idx = 3

        # Extract accelerator data.
        with open(acc_path, "rb") as f: 
            acc_data = pickle.load(f)  # [key, path, label, feature_array] 

        self.acc_label_list = []
        self.acc_feature_list = []
        for data in acc_data:
            self.acc_label_list.append(data[label_idx])
            self.acc_feature_list.append(data[feature_idx].T)

        self.acc_feature_list = torch.tensor(np.float32(np.array(self.acc_feature_list)))

        # Extract gyro data.
        with open(gyro_path, "rb") as f: 
            gyro_data = pickle.load(f)  # [key, path, label, feature_array] 

        self.gyro_label_list = []
        self.gyro_feature_list = []
        for data in gyro_data:
            self.gyro_label_list.append(data[label_idx])
            self.gyro_feature_list.append(data[feature_idx].T)

        self.gyro_feature_list = torch.tensor(np.float32(np.array(self.gyro_feature_list)))
        
        # Sanity check.
        assert self.acc_label_list == self.gyro_label_list, 'The labels from both accelerator and gyroscope should be the same.'

        self.classes = list(set(self.acc_label_list))

    def __len__(self):
        return len(self.acc_label_list)

    def __getitem__(self, idx):
        acc_feature = self.acc_feature_list[idx]
        gyro_feature = self.gyro_feature_list[idx]
        acc_label = self.acc_label_list[idx]
        if self.return_label:
            return acc_feature, gyro_feature, acc_label
        return acc_feature, gyro_feature

