import os
import lmdb
import pickle
import torch
import numpy as np
import selfies as sf
from typing import List
from rdkit import Chem
from torch.utils.data import Dataset

from helper.register import DATASETS
from data.augment import IRTransformBase

# 数据类
@DATASETS.register_module(force=True)
class IRDataset(Dataset):
    def __init__(self, db_path, transforms: List[IRTransformBase] = []):
        self.this_dir = os.path.dirname(__file__)
        self.db_path = os.path.abspath(os.path.join(self.this_dir, '..', db_path))
        self.transforms = transforms
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, item):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        key = self._keys[item]
        datapoint_pickled = self.env.begin().get(key)
        data = pickle.loads(datapoint_pickled)
        if "smi" in data.keys():
            # COC12CC3C4C3C1C24
            data['gt_smi'] = data['smi']
            data["smi"] = self.get_canonical_smile(data["smi"])
            data['mode'] = 'train'
        else:
            data['mode'] = 'test'
        if 'ir' in data.keys():
            data['ir'] = data['ir']
        data = self.forward(data)
        return data

    def forward(self, data):
        for t in self.transforms:
            data = t.forward(data)
        return data

    def invert(self, data):
        for t in self.transforms:
            data = t.invert(data)
        return data

    @staticmethod
    def get_canonical_smile(testsmi):
        try:
            mol = Chem.MolFromSmiles(testsmi)
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            print("Cannot convert {} to canonical smiles")
            return testsmi


def get_selfies(smi):
    try:
        return sf.encoder(smi)
    except:
        mol = Chem.MolFromSmiles(smi)
        smiles = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)
        return sf.encoder(smiles)



class MyCollator(object):
    def __init__(self, device, tokenizer, max_length=512, ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __call__(self, examples):
        input = {}
        smi = []
        ir = []
        gt_smi = []
        for i in examples:
            gt_smi.append(i['smi'])
            smi.append(get_selfies(i["smi"]))
            ir.append(i["ir"])
        ir = np.ascontiguousarray(np.array(ir))
        if len(smi) > 0:
            output = self.tokenizer(
                smi,
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            # attention_mask, input_ids, ...
            input["labels"] = output["input_ids"].contiguous().to(self.device)
            input["labels"] = torch.where(input['labels'] != self.tokenizer.pad_token_id, input['labels'], -100)
        input["ir"] = torch.from_numpy(ir).float().to(self.device)
        input['gt_smi'] = gt_smi
        return input


if __name__ == '__main__':
    dataset = IRDataset('qm9/temp.mdb')


