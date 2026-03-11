import os.path as op
import random
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset

import os
import json
import glob
from prettytable import PrettyTable
import collections
import numpy as np

class WebREID(BaseDataset):
    dataset_dir = ''
    def __init__(self,args='', root='', verbose=True):
        super(WebREID, self).__init__()

        self.dataset_rewrite_dir = '/mnt/tianluzheng/dataset/WebReid'
        json_data_all = []
        json_data_all.append('/mnt/tianluzheng/dataset/WebReid/100w.json')
        self.train_img_paths = []
        for json_file in json_data_all:
            input_path = os.path.join(json_file)
            with open(input_path) as f:
                data = json.load(f)
            # part = 'qwen_person'
            for item in data:
                for key,value in item.items():
                    key = os.path.join(self.dataset_rewrite_dir,'images/',key)
                    self.train_img_paths.append(key)
        
        train_cap_dict = self._merged_multi_json_file(args,json_data_all)

        self.train, self.train_id_container, self.part_dataset, num_caption,self.fpath2part_cap,self.fpaht2sim = self._get_dataset(self.train_img_paths, train_cap_dict)

        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(['train', len(set(self.train_id_container)),len(self.train), num_caption])
        # table.add_row(['test', len(self.test["image_pids"]),len(self.test["image_pids"]), len(self.test["image_pids"])])
        self.logger.info('\n' + str(table))
    
    def read_json_files(self,directory):
        path_pattern = os.path.join(directory, '**', 'BLIP2', '*.json')

        json_files = glob.glob(path_pattern, recursive=True)
        return json_files    

    def _merged_json_file(self, json_path_list):
        merged_dict = {}

        for file_path in json_path_list:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                merged_dict.update(data)
        return merged_dict
    
    def _merged_multi_json_file(self, args,json_path_list):
        merged_dict = collections.defaultdict(list)

        part0 = 'caption1'
        part = 'caption2'
        for file_path in json_path_list:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                print(file_path, len(data))
                for item in data:
                    for k,value in item.items():
                        key = os.path.join(self.dataset_rewrite_dir,'images/',k)
                        merged_dict[key].append(value[part0].replace("/n",""))
                        # if len(value[part].split())> 5: 
                        merged_dict[key].append(value[part].replace("/n",""))
                        # print(k)
        return merged_dict

    def _get_test_dataset(self, test_img_paths, cap_dict):
        dataset = {}
        img_paths = []
        captions = []
        image_pids = []
        caption_pids = []
        for i in range(len(test_img_paths)):
            pid = i
            img_path = test_img_paths[i]
            img_paths.append(img_path)
            image_pids.append(pid)
            path2cap = '/'.join(img_path.split('/')[-1])
            caption = cap_dict[path2cap][0]
            captions.append(caption)
            caption_pids.append(pid)
        dataset = {
            "image_pids": image_pids,
            "img_paths": img_paths,
            "caption_pids": caption_pids,
            "captions": captions
        }
        return dataset
    
    def _get_dataset(self, img_paths, cap_dict):
        pid_container = set()
        img_paths = sorted(img_paths)

        dataset = []
        part_dataset = []
        idx_count = 0
        pid_count = 0
        num_caption = 0

        fpath2part_cap = {}
        fpaht2sim = {}
        for i in range(len(img_paths)):
            img_path = img_paths[i]
            caption = cap_dict[img_path]

            fpath2part_cap[img_path] = {}
            fpaht2sim[img_path] = {}
            pid = pid_count
            image_id = idx_count
            pid_container.add(pid)
            for cap in caption:
                part2sim = 77 * [1- 0.15]
                part2sim = np.array(part2sim)
                dataset.append([pid,idx_count,img_path, cap, part2sim])# , part2sim
                num_caption += 1
                idx_count += 1
                pid_count += 1
        assert idx_count == len(dataset)

        return dataset, pid_container, part_dataset,num_caption,fpath2part_cap,fpaht2sim