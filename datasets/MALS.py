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

class MALS_structure(BaseDataset):
    dataset_dir = ''
    def __init__(self,args='', root='', verbose=True):
        super(MALS_structure, self).__init__()
        #过滤后的数据
        self.dataset_dir = '/mnt/tianluzheng/PersonReID_dataset/MALS/gene_attrs'
        self.dataset_image='/mnt/tianluzheng/PersonReID_dataset/MALS'
        sub_dirs=[]
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith('.json'):  # 确保只处理JSON文件
                sub_dirs.append(filename)
        json_data_all = []
        for file_name in sub_dirs:
            file_path=os.path.join(self.dataset_dir,file_name)
            json_data_all.append(file_path)
            print(f"Read {len(json_data_all)} JSON files from {self.dataset_dir}")
        self.train_img_paths = []
        # self.train_cap = []
        for json_file in json_data_all:
            input_path = os.path.join(json_file)
            with open(input_path) as f:
                data = json.load(f)
            # part = 'qwen_person'
            for i in range (len(data)):
                image_path = data[i]['image']
                key = os.path.join(self.dataset_image,image_path)
                self.train_img_paths.append(key)
        
        train_cap_dict = self._merged_multi_json_file(json_data_all)
        # test_cap_dict = self._merged_json_file(self.test_cap_paths)

        self.train, self.train_id_container, self.part_dataset, num_caption,self.fpath2part_cap,self.fpaht2sim = self._get_dataset(self.train_img_paths, train_cap_dict)
        # self.test = self._get_test_dataset(self.test_img_paths, test_cap_dict)
        
        self.logger.info("=> BLIP2 Images and Captions are loaded")
        self.logger.info("BLIP2 Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(['train', len(set(self.train_id_container)),len(self.train), num_caption])
        # table.add_row(['test', len(self.test["image_pids"]),len(self.test["image_pids"]), len(self.test["image_pids"])])
        self.logger.info('\n' + str(table))
    
    def read_json_files(self,directory):
        # 构建用于查找.json文件的路径模式
        path_pattern = os.path.join(directory, '**', 'BLIP2', '*.json')

        # 使用glob.glob遍历所有匹配的文件路径
        json_files = glob.glob(path_pattern, recursive=True)

        return json_files    

    def _merged_json_file(self, json_path_list):
        merged_dict = {}

        # 逐个读取JSON文件并合并到字典中
        for file_path in json_path_list:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                merged_dict.update(data)
        return merged_dict
    
    def _merged_multi_json_file(self, json_path_list):
        merged_dict = collections.defaultdict(list)

        # part0 = 'qwen_person'
        # part = 'llama_rewrite'#qwen_person
        for file_path in json_path_list:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                print(file_path, len(data))
            for i in range(len(data)):
                image_path = data[i]['image']
                key = os.path.join(self.dataset_image,image_path)
                caption = data[i]['caption']
                merged_dict[key].append(caption)
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
            
            # path2cap = img_path.split('/')[-1]
            caption = cap_dict[img_path]

            # if len(caption) != 4:
            #     continue
            fpath2part_cap[img_path] = {}
            fpaht2sim[img_path] = {}
            pid = pid_count
            image_id = idx_count
            pid_container.add(pid)
            for cap in caption:
                part2sim = 77 * [1- 0.15]
                part2sim = np.array(part2sim)
                dataset.append([pid,idx_count,img_path, cap, part2sim])
                num_caption += 1
                idx_count += 1
                pid_count += 1
        assert idx_count == len(dataset)

        return dataset, pid_container, part_dataset,num_caption,fpath2part_cap,fpaht2sim