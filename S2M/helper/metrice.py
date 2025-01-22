import numpy as np
from rdkit import Chem

def same_smi(smi1, smi2):
    try:
        key1 = Chem.MolToInchiKey(Chem.MolFromSmiles(smi1))
        key2 = Chem.MolToInchiKey(Chem.MolFromSmiles(smi2))
        return key1 == key2 and key1 is not None
    except:
        return False


def topK_metric(gts, topk_preds, plain=False, reduction='sum_weighted'):
    """
    reduction
        - raw:
        - raw_weighted:
        - sum:
        - sum_weighted:
    """
    try:
        weights = np.array([0.4, 0, 0.1, 0, 0.1, 0, 0, 0, 0, 0.4])
        hits = np.zeros((len(topk_preds), len(topk_preds[0])))
        for idx, (gt, topk_pred) in enumerate(zip(gts, topk_preds)):
            for i, pred in enumerate(topk_pred):
                if not plain:
                    match_func = same_smi
                else:
                    match_func = lambda x, y: x == y
                if match_func(gt, pred):
                    hits[idx, i:] = 1
                    break

        # vf = np.vectorize(same_smi)
        # hits = vf(np.array(gts).reshape(-1, 1), topk_preds)

        score = np.mean(hits, axis=0)
        if reduction.endswith('weighted'):
            score = score * weights[:hits.shape[1]]
        if reduction.startswith('raw'):
            return score.tolist()
        elif reduction.startswith('sum'):
            return score.sum().item()
    except:
        return 0