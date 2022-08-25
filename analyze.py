import pandas as pd
import os
from tqdm import tqdm
import utils

if __name__ == '__main__':
    path = './inferences'
    vocabs = []
    variances = []
    radius = []
    pbar = tqdm(os.listdir(path))
    for class_name in pbar:
        vocab, features, total = utils.load_features(os.path.join(path, class_name))
        var, r = utils.compute_variance(features)
        vocabs.append(vocab)
        variances.append(var)
        radius.append(r)

    data = {
        'vocab': vocabs,
        'variance': variances,
        'radius': radius
    }
    df = pd.DataFrame(data)
    df.to_csv('./output/result.csv', index=False)



