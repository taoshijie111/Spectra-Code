import numpy as np
import scipy.ndimage as ndi
import random
from rdkit import Chem
from rdkit.Chem import Fragments, rdMolDescriptors
from collections import OrderedDict

from helper.register import TRANSFORMS


class IRTransformBase:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

    def forward(self, data: dict):
        raise NotImplementedError

    def invert(self, data: dict):
        return data


"""
IR 增强
"""


@TRANSFORMS.register_module()
class IRSpectrSOS(IRTransformBase):
    """ smooth or sharpen
    gaussian filter 1d
    """

    def __init__(self, sigma=3.0):
        self.sigma = float(sigma)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(sigma={self.sigma})'

    def forward(self, data):
        if 'ir' in data:
            sigma = np.random.uniform(-self.sigma, self.sigma)
            raw = data['ir']
            blurred = ndi.gaussian_filter1d(raw, abs(sigma))
            data['ir'] = sigma
            if sigma > 0:
                # print('smooth', sigma)
                data['ir'] = blurred
            else:
                # print('sharpen', sigma)
                data['ir'] = raw + (raw - blurred)
        return data


@TRANSFORMS.register_module()
class IRSpectrShift(IRTransformBase):
    """
    平移光谱
    """
    def __init__(self, shift=5):
        self.shift = int(shift)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(shift={self.shift})'

    def forward(self, data):
        if 'ir' in data:
            shift = np.random.randint(-self.shift, self.shift)
            data['ir_shift'] = shift
            data['ir'] = ndi.shift(data['ir'], shift, mode='constant')
        return data

    def invert(self, data):
        if 'ir_shift' in data:
            data['ir'] = ndi.shift(data['ir'], - data.pop('ir_shift'), mode='constant')
        return data


@TRANSFORMS.register_module()
class IRSpectrScale(IRTransformBase):
    """
    对光谱缩放
    """
    def __init__(self, scale=0.001):
        self.scale = float(scale)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(scale={self.scale})'

    def forward(self, data):
        if 'ir' in data:
            scale = np.clip(np.random.randn(), -3, 3) * self.scale
            data['ir_scale'] = scale
            data['ir'] = data['ir'] * (1 + scale)
        return data

    def invert(self, data):
        if 'ir_scale' in data:
            data['ir'][:, 1] = data['ir'][:, 1] / (1 + data.pop('ir_scale'))
        return data


@TRANSFORMS.register_module()
class IRSpectrSelect(IRTransformBase):
    """
    选择某个范围内的光谱，其余为0
    """
    def __init__(self, select=[10, 2000]):
        self.select = select
        assert len(select) % 2 == 0

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(select={self.select})'

    def forward(self, data):
        if 'ir' in data:
            if self.select is not None:
                new_ir_data = np.zeros_like(data['ir'])
                freq = np.arange(len(data['ir']))
                for i in range(0, len(self.select), 2):
                    s, e = self.select[i], self.select[i + 1]
                    idx = np.nonzero((s <= freq) * (freq < e))
                    new_ir_data[idx] = data['ir'][idx]
                data['ir'] = new_ir_data
        return data

    def invert(self, data):
        return data


@TRANSFORMS.register_module()
class IRNorm(IRTransformBase):
    """
    光谱归一化，ir / intensity_max
    """
    def __init__(self, intensity_max=None) -> None:
        super().__init__()
        self.intensity_max = intensity_max

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(intensity_max={self.intensity_max})'

    def forward(self, data):
        if 'ir' in data:
            if self.intensity_max is None:
                intensity_max = np.max(data['ir'])
            else:
                intensity_max = self.intensity_max
            data['ir'] /= intensity_max
            data['intensity_max'] = intensity_max
        return data

    def invert(self, data):
        if 'intensity_max' in data:
            data['ir'] *= data.pop('intensity_max')
        return data


@TRANSFORMS.register_module()
class IRLog(IRTransformBase):
    """
    对光谱数据取log
    """
    def __init__(self, base='log2') -> None:
        super().__init__()
        self.base = getattr(np, base)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(base={self.base})'

    def forward(self, data):
        if 'ir' in data:
            data['ir'] = np.log2(data['ir'] + 1) / 10
        return data

    def invert(self, data):
        if 'ir' in data:
            data['ir'] = np.power(2, 10 * data.pop('ir')) - 1
        return data

"""
smiles 增强
"""


@TRANSFORMS.register_module()
class IRSmiles(IRTransformBase):
    def __init__(self, prob=1.0) -> None:
        super().__init__()
        self.prob = float(prob)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(prob={self.prob})'

    def forward(self, data):
        if 'smi' in data:
            if self.prob >= 1 or random.random() < self.prob:
                m1 = Chem.MolFromSmiles(data['smi'])
                m1.SetProp("_canonicalRankingNumbers", "True")
                idxs = list(range(0, m1.GetNumAtoms()))
                random.shuffle(idxs)
                for i, v in enumerate(idxs):
                    m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
                data['smi'] = Chem.MolToSmiles(m1)
        return data


@TRANSFORMS.register_module()
class IRFormula(IRTransformBase):
    def forward(self, data):
        if 'smi' in data:
            if 'formula' in data:
                return data
            mol = Chem.MolFromSmiles(data['smi'])
            formula = rdMolDescriptors.CalcMolFormula(mol)
            data['formula'] = formula
        return data


@TRANSFORMS.register_module()
class IRFuncGroups(IRTransformBase):
    functional_groups = OrderedDict({
        "Alcohol": Chem.MolFromSmarts("[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15])]"),
        # "Carboxylic Acid": Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
        "Ester": Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]"),
        "Ether": Fragments.fr_ether,
        "Aldehyde": Chem.MolFromSmarts("[CX3H1](=O)[#6]"),
        "Ketone": Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
        "Alkene": Chem.MolFromSmarts("[CX3]=[CX3]"),
        "Alkyne": Chem.MolFromSmarts("[$([CX2]#C)]"),
        "Benzene": Fragments.fr_benzene,
        "Primary Amine": Chem.MolFromSmarts("[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]"),
        "Secondary Amine": Fragments.fr_NH1,
        "Tertiary Amine": Fragments.fr_NH0,
        "Amide": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]"),
        "Cyano": Chem.MolFromSmarts("[NX1]#[CX2]"),
        "Fluorine": Chem.MolFromSmarts("[#6][F]"),
        # "Chlorine": Chem.MolFromSmarts("[#6][Cl]"),
        # "Iodine": Chem.MolFromSmarts("[#6][I]"),
        # "Bromine": Chem.MolFromSmarts("[#6][Br]"),
        # "Sulfonamide": Chem.MolFromSmarts("[#16X4]([NX3])(=[OX1])(=[OX1])[#6]"),
        # "Sulfone": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[#6]"),
        # "Sulfide": Chem.MolFromSmarts("[#16X2H0]"),
        # "Phosphoric Acid": Chem.MolFromSmarts(
        #     "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]"
        # ),
        # "Phosphoester": Chem.MolFromSmarts(
        #     "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]"
        # ),
    })
    num_func_groups = len(functional_groups)

    @staticmethod
    def match_group(mol: Chem.Mol, func_group) -> int:
        if type(func_group) == Chem.Mol:
            n = len(mol.GetSubstructMatches(func_group))
        else:
            n = func_group(mol)
        return 0 if n == 0 else 1

    def forward(self, data):
        if "smi" in data:
            if 'func_groups' in data:
                return data
            mol = Chem.MolFromSmiles(data['smi'])
            func_groups = OrderedDict()
            for func_group_name, smarts in self.functional_groups.items():
                func_groups[func_group_name] = self.match_group(mol, smarts)
            data['func_groups'] = ''.join([str(func_groups[k]) for k in func_groups])
        return data


@TRANSFORMS.register_module()
class IRCollect(IRTransformBase):
    def __init__(self, keys=['smi']) -> None:
        if 'smi' in keys:
            keys.remove('smi')
            keys.append('smi')
        self.keys = keys

    def forward(self, data: dict):
        if 'smi' in data:
            if 'smi' not in self.keys:
                data['smi_bakup'] = data['smi']
            data['smi'] = '+'.join([str(data[key]) for key in self.keys])
        return data


if __name__ == '__main__':
    from data.QM9 import IRDataset
    from copy import deepcopy
    import matplotlib.pyplot as plt
    import matplotlib
    import pickle

    matplotlib.use('TkAgg')

    def draw_ir(ir, ir_aug):
        x = np.arange(1, len(ir)+1)
        y = ir
        y_aug = ir_aug

        plt.plot(x, y, 'b', label='original')
        plt.plot(x, y_aug, 'r', label='augmented')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Curve Plot')
        plt.savefig('curve_plot.png')
        plt.legend()
        plt.show()

    with open('../qm9/temp_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    print(len(dataset))
    data = dataset[0]
    transformer = IRSpectrSOS()
    data_aug = transformer.forward(deepcopy(data))
    draw_ir(data['ir'], data_aug['ir'])
