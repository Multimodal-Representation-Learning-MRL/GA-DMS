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
import pyarrow as pa


class LUPersonT(BaseDataset):
    dataset_dir = ''
    def __init__(self, args= '',root='', verbose=True):
        super(LUPersonT, self).__init__()
        # #过滤后的数据
        self.dataset_image='/mnt/tianluzheng/PersonReID_dataset/LUPerson-T/LUPerson_clip_128w_phrase_arrow/luperson-t-image'
        with open('/mnt/tianluzheng/PersonReID_dataset/LUPerson-T/LUPerson_clip_128w_phrase_arrow/LUPerson_train.arrow', 'rb') as f:
            reader = pa.ipc.open_file(f)
            table = reader.read_all()

        # 将 Arrow Table 转换为 Pandas DataFrame，如果需要的话
        df = table.to_pandas()



        self.train, self.train_id_container, self.part_dataset, num_caption,self.fpath2part_cap,self.fpaht2sim = self._get_dataset(df)
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

    
    def _get_dataset(self, df):
        pid_container = set()
        # img_paths = sorted(img_paths)

        dataset = []
        part_dataset = []
        idx_count = 0
        pid_count = 0
        num_caption = 0

        fpath2part_cap = {}
        fpaht2sim = {}
        captions = df["caption"]

        for i in range(len(captions)-100000):
            caption= captions[i]
            # print(captions)
            # 读取i.jpg
            img_path = os.path.join(self.dataset_image,f'{i}.jpg')

            fpath2part_cap[img_path] = {}
            fpaht2sim[img_path] = {}
            pid = pid_count
            image_id = idx_count
            pid_container.add(pid)
            for cap in caption:
                # if 'description]' in cap or '<' in cap or '>'in cap or '/' in cap or 'jekyll' in cap:
                #     self.logger.info(cap)
                #     break 
                    # try:
                    #     cap = random.choice(safe_dict[img_path])
                    # except:
                    #     pass
                print(cap)
                part2sim = 77 * [1- 0.15]
                part2sim = np.array(part2sim)
                dataset.append([pid,idx_count,img_path, cap, part2sim])
                num_caption += 1
                idx_count += 1
                pid_count += 1
        assert idx_count == len(dataset)

        return dataset, pid_container, part_dataset,num_caption,fpath2part_cap,fpaht2sim