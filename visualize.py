import os

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import utils

if __name__ == '__main__':
    path = './output/result.csv'
    df = pd.read_csv(path)
    df['distance'] = 0.7*df['variance'] + 0.3*df['radius']
    # df_radius = df.sort_values(by='radius', ascending=False)
    # df_var = df.sort_values(by='variance', ascending=False)
    df_temp = df.sort_values(by='distance',ascending=False).reset_index()
    # print(df_radius.head(5))
    # print(df_var.head(5))
    vocabs = df_temp.head(5)['vocab'].map(lambda x: x.split('\\')[-1]).tolist()
    print(vocabs)

    colors = ['blue', 'red', 'green', 'cyan',' yellow']
    feature_path = './inferences'
    data = {}
    pca = PCA(n_components=3)
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i,vocab in enumerate(vocabs):
        voc, feat, total = utils.load_features(os.path.join(feature_path, vocab), n=1000)
        feat_new = pca.fit_transform(feat)
        ax.scatter(feat_new[:,0], feat_new[:,1],feat_new[:,2], cmap=colors[i])
        data[vocab] = feat_new
    plt.show()


