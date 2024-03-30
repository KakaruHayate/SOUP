import os
import numpy as np
import librosa
import torch
import pyworld as pw
import argparse
import shutil
from logger import utils
from tqdm import tqdm
from logger.utils import traverse_dir
from modules.rmvpe import RMVPE
from modules.SOME import *
import concurrent.futures
import soundfile as sf

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)
    
def preprocess(
        path_srcdir, 
        path_f0dir,
        path_spdir,
        path_mididir,
        device,
        f0_min,
        f0_max,
        sampling_rate,
        hop_length, 
        model_rmvpe, 
        SOME_ext
        ):
        
    # list files
    filelist =  traverse_dir(
        path_srcdir,
        extension='wav',
        is_pure=True,
        is_sort=True,
        is_ext=True)


    # run
    
    def process(file):
        ext = file.split('.')[-1]
        binfile = file[:-(len(ext)+1)]+'.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_spfile = os.path.join(path_spdir, binfile)
        path_f0file = os.path.join(path_f0dir, binfile)
        path_midifile = os.path.join(path_mididir, binfile)
        
        # load audio
        #x, sr = sf.read(path_srcfile)
        x, _ = librosa.load(path_srcfile, sr=sampling_rate)
        x = x.astype(np.double)
        
        # get length using dio
        _, t = pw.dio(
            x, 
            sampling_rate, 
            f0_floor=f0_min, 
            f0_ceil=f0_max, 
            channels_in_octave=2, 
            frame_period=(1000*hop_length / sampling_rate))
        
        # extract f0 using rmvpe
        f0 = model_rmvpe.infer_from_audio(x, sample_rate=sampling_rate, thred=0.03, use_viterbi=False)
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        uv = np.interp(np.linspace(0, 1, len(t)), np.linspace(0, 1, len(f0)), uv.astype(float)) > 0.5
        f0 = np.interp(np.linspace(0, 1, len(t)), np.linspace(0, 1, len(f0)), f0)
        f0[uv] = 0
        
        
        # extract spectral envelope using cheaptrick
        
        sp = pw.cheaptrick(x, f0, t, sampling_rate)
        
        
        # extract midi using SOME
        midi_res = SOME_ext.inference(wf = x, osr = sampling_rate)
        
        
        
        f0 = f0.astype(float)
        sp = sp.astype(float)
        midi = expand_midi(midi_res, f0)
        midi = midi.astype(float)
        
        os.makedirs(os.path.dirname(path_spfile), exist_ok=True)
        np.save(path_spfile, sp)
        os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
        np.save(path_f0file, f0)
        os.makedirs(os.path.dirname(path_midifile), exist_ok=True)
        np.save(path_midifile, midi)

    print('Preprocess the audio clips in :', path_srcdir)
    
    # single process
    for file in tqdm(filelist, total=len(filelist)):
        process(file)
    
    # multi-process (have bugs)
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(process, filelist), total=len(filelist)))
    '''
                
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_rmvpe = RMVPE("pretrain/rmvpe/model.pt", hop_length=160, device=device)
    SOME_ext = SOME(some_path = "pretrain/SOME/0918_continuous128_clean_3spk.onnx")
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    f0_min = args.data.f0_min
    f0_max = args.data.f0_max
    sampling_rate  = args.data.sampling_rate
    hop_length = args.data.block_size
    train_path = args.data.train_path
    valid_path = args.data.valid_path
    
    # run
    for path in [train_path, valid_path]:
        path_srcdir  = os.path.join(path, 'audio')
        path_spdir  = os.path.join(path, 'sp')
        path_f0dir  = os.path.join(path, 'f0')
        path_mididir  = os.path.join(path, 'midi')
        preprocess(
            path_srcdir, 
            path_f0dir,
            path_spdir,
            path_mididir,
            device,
            f0_min,
            f0_max,
            sampling_rate,
            hop_length,
            model_rmvpe,
            SOME_ext)
    
