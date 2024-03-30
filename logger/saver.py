'''
author: wayn391@mastertones
'''

import os
import json
import time
import yaml
import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
from . import utils
from torch.utils.tensorboard import SummaryWriter
from matplotlib.ticker import MultipleLocator

class Saver(object):
    def __init__(
            self, 
            args,
            initial_global_step=-1):

        self.expdir = args.env.expdir
        self.sample_rate = args.data.sampling_rate
        
        # cold start
        self.global_step = initial_global_step
        self.init_time = time.time()
        self.last_time = time.time()

        # makedirs
        os.makedirs(self.expdir, exist_ok=True)       

        # path
        self.path_log_info = os.path.join(self.expdir, 'log_info.txt')

        # ckpt
        os.makedirs(self.expdir, exist_ok=True)       

        # writer
        self.writer = SummaryWriter(os.path.join(self.expdir, 'logs'))
        
        # save config
        path_config = os.path.join(self.expdir, 'config.yaml')
        with open(path_config, "w") as out_config:
            yaml.dump(dict(args), out_config)


    def log_info(self, msg):
        '''log method'''
        if isinstance(msg, dict):
            msg_list = []
            for k, v in msg.items():
                tmp_str = ''
                if isinstance(v, int):
                    tmp_str = '{}: {:,}'.format(k, v)
                else:
                    tmp_str = '{}: {}'.format(k, v)

                msg_list.append(tmp_str)
            msg_str = '\n'.join(msg_list)
        else:
            msg_str = msg
        
        # dsplay
        print(msg_str)

        # save
        with open(self.path_log_info, 'a') as fp:
            fp.write(msg_str+'\n')

    def log_value(self, dict):
        for k, v in dict.items():
            self.writer.add_scalar(k, v, self.global_step)
    
    def log_audio(self, dict):
        for k, v in dict.items():
            self.writer.add_audio(k, v, global_step=self.global_step, sample_rate=self.sample_rate)
    
    def log_f0(self, name, pref0, gtf0):
        exp_img_dir = 'exp/img'
        if not os.path.exists(exp_img_dir):
            os.makedirs(exp_img_dir)
        if pref0.is_cuda:
            pref0 = pref0.cpu()
        if gtf0.is_cuda:
            gtf0 = gtf0.cpu()
        if isinstance(pref0, torch.Tensor):
            pref0 = pref0.squeeze().numpy()
        if isinstance(gtf0, torch.Tensor):
            gtf0 = gtf0.squeeze().numpy()
        pref0 = np.where(pref0 < 1e-3, 0, np.round(pref0, 4))
        fig = plt.figure(figsize=(12, 9))

        plt.plot(pref0, color='blue', label='predicted f0')
        plt.plot(gtf0, color='red', label='ground truth f0')
        plt.title(name)
        plt.xlabel('time')
        plt.ylabel('f0')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        #plt.show()
        image_file = os.path.join(exp_img_dir, f'{name}_{self.global_step}.png')
        plt.savefig(image_file)
        plt.close(fig)
        #self.writer.add_figure(name, figure=fig, global_step=self.global_step)
    
    def get_interval_time(self, update=True):
        cur_time = time.time()
        time_interval = cur_time - self.last_time
        if update:
            self.last_time = cur_time
        return time_interval

    def get_total_time(self, to_str=True):
        total_time = time.time() - self.init_time
        if to_str:
            total_time = str(datetime.timedelta(
                seconds=total_time))[:-5]
        return total_time

    def save_model(
            self,
            model, 
            optimizer,
            name='model',
            postfix='',
            to_json=False):
        # path
        if postfix:
            postfix = '_' + postfix
        path_pt = os.path.join(
            self.expdir , name+postfix+'.pt')
       
        # check
        print(' [*] model checkpoint saved: {}'.format(path_pt))

        # save
        torch.save({
            'global_step': self.global_step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}, path_pt)
            
        # to json
        if to_json:
            path_json = os.path.join(
                self.expdir , name+'.json')
            utils.to_json(path_params, path_json)
            
    def global_step_increment(self):
        self.global_step += 1


