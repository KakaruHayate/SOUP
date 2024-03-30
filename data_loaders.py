import os
import random
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset

def traverse_dir(
        root_dir,
        extension,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args, whole_audio=False):
    data_train = AudioDataset(
        args.data.train_path,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate
        )
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.train.num_workers,
        persistent_workers=(args.train.num_workers > 0),
        pin_memory=True
    )
    data_valid = AudioDataset(
        args.data.valid_path,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate
        )
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return loader_train, loader_valid 


class AudioDataset(Dataset):
    def __init__(
        self,
        path_root,
        hop_size,
        sample_rate
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extension='wav',
            is_pure=True,
            is_sort=True,
            is_ext=False
        )
        self.data_buffer={}
        print('Load all the data from :', path_root)
        for name in tqdm(self.paths, total=len(self.paths)):
            
            path_f0 = os.path.join(self.path_root, 'f0', name) + '.npy'
            f0 = np.load(path_f0)
            #f0 = f0.reshape(1,f0.shape[0])
            f0 = torch.from_numpy(f0).float()
                
            path_sp = os.path.join(self.path_root, 'sp', name) + '.npy'
            sp = np.load(path_sp)
            #sp = sp.reshape(1,sp.shape[0],sp.shape[1])
            sp = torch.from_numpy(sp).float()
            
            path_midi = os.path.join(self.path_root, 'midi', name) + '.npy'
            midi = np.load(path_midi)
            midi = midi.reshape(midi.shape[0],1)
            midi = torch.from_numpy(midi).float()
            
            self.data_buffer[name] = {
                    'sp': sp,
                    'f0': f0,
                    'midi': midi
                    }
           

    def __getitem__(self, file_idx):
        name = self.paths[file_idx]
        data_buffer = self.data_buffer[name]

        # get item
        return self.get_data(name, data_buffer)

    def get_data(self, name, data_buffer):
        # load f0
        f0 = data_buffer.get('f0')
        
        # load sp
        sp = data_buffer.get('sp')
        
        # load midi
        midi = data_buffer.get('midi')
        
        return dict(f0=f0, sp=sp, midi=midi, name=name)

    def __len__(self):
        return len(self.paths)
