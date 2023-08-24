'''
Chandio, B.Q., Chattopadhyay, T., Owens-Walton, C., Reina, J.E.V., 
Nabulsi, L., Thomopoulos, S.I., Garyfallidis, E. and Thompson, P.M., 2022. 
FiberNeat: Unsupervised White Matter Tract Filtering. bioRxiv.

Adapted by Wendy Feng (07/31/2023)
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from dipy.tracking.streamline import set_number_of_points
from sklearn.cluster import DBSCAN
import seaborn as sns
from dipy.segment.metric import mdf
import time

import argparse
from dipy.io.streamline import load_trk
from utils import save_bundle


def _FiberNeat_tSNE(bun1, dist, start, p=None, k=None):
    
    n = len(bun1)
    y = np.ones(n)

    p = n*.065

    if n < 800:
        p = n*.25

    if n > 4000:
        p = n*.02

    X_embedded = TSNE(n_components=2, init='random', metric='precomputed', perplexity=p).fit_transform(dist)

    tsne_result_df = pd.DataFrame({'tsne_1': X_embedded[:,0],
                                   'tsne_2': X_embedded[:,1],
                                   'label': y})

    
    k = p*.007

    clustering = DBSCAN(eps=k, min_samples=2).fit(tsne_result_df)
    l = clustering.labels_
    dbtsne_result_df = pd.DataFrame({'tsne_1': X_embedded[:,0],
                                   'tsne_2': X_embedded[:,1],
                                   'label': l})
    
    
    lb = list(set(clustering.labels_))
    big_c = 0
    size = 0
    for i in lb:
        if len(bun1[l==i])>size:
            big_c = i
            size = len(bun1[l==i])
            
    bun2 = bun1[l==big_c]
    
    print("time taken in seconds = ", time.time()-start)
    
    # below is just plotting and not part of core method so not counting it in time taken by the method
        
    
    cmap = plt.cm.get_cmap('Reds')
    plt.title("Streamline distances")
    plt.imshow(dist, cmap=cmap)
    plt.colorbar()
    
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
    lim = (X_embedded.min()-5, X_embedded.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.set_title("t-SNE embedding")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=dbtsne_result_df, ax=ax,s=120)
    lim = (X_embedded.min()-5, X_embedded.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.set_title("DBSCAN Clustering")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    return l==big_c
    
    
def _FiberNeat_UMAP(bun1, dist, start, p=None, k=None):
    
    n = len(bun1)
    y = np.ones(n)

    p = int(n*0.05)

    mapper = umap.UMAP(n_neighbors=p).fit(dist)
    low_dim_emb_train = mapper.transform(dist)

    umap_result_df = pd.DataFrame({'umap_1': low_dim_emb_train[:,0],
                                   'umap_2': low_dim_emb_train[:,1],
                                   'label': y})

    
    k = p*.0025
    if n < 800:
        k = 1

    clustering = DBSCAN(eps=k, min_samples=2).fit(umap_result_df)
    l = clustering.labels_
    dbumap_result_df = pd.DataFrame({'umap_1': low_dim_emb_train[:,0],
                                   'umap_2': low_dim_emb_train[:,1],
                                   'label': l})


    
    
    lb = list(set(clustering.labels_))
    big_c = 0
    size = 0
    for i in lb:
        if len(bun1[l==i])>size:
            big_c = i
            size = len(bun1[l==i])
            
    bun2 = bun1[l==big_c]
    
    
    print("time taken in seconds = ", time.time()-start)
    
    # below is just plotting and not part of core method so not counting it in time taken by the method
    cmap = plt.cm.get_cmap('Reds')
    plt.title("Streamline distances")
    plt.imshow(dist, cmap=cmap)
    plt.colorbar()
    
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='umap_1', y='umap_2', hue='label', data=umap_result_df, ax=ax,s=120)
    lim = (low_dim_emb_train.min()-5, low_dim_emb_train.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.set_title("UMAP embedding")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='umap_1', y='umap_2', hue='label', data=dbumap_result_df, ax=ax,s=120)
    lim = (low_dim_emb_train.min()-5, low_dim_emb_train.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.set_title("DBSCAN Clustering")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    
    return l==big_c
    
    
def FiberNeat(bundle_in, dim_method='tsne', p=None, k=None):
    
    start = time.time()
    
    bundle = set_number_of_points(bundle_in,20)
    
    n = len(bundle)
    dist = np.zeros((n,n))

    for i in range(n):

        s1 = bundle[i]

        for j in range(n):

            s2 = bundle[j]

            dist[i][j] = mdf(s1,s2)


    
    if dim_method.lower()=='tsne':
        cleaned_bundle = _FiberNeat_tSNE(bundle, dist, start)
    elif dim_method.lower()=='umap':
        cleaned_bundle = _FiberNeat_UMAP(bundle, dist, start)
    else:
        print("invalid dimentionality reduction method, ", dim_method)
        
    
    return bundle_in[cleaned_bundle]

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', '-i', type=str, required=True)
    parser.add_argument('--outpath', '-o', type=str, required=True)
    parser.add_argument('--dim_method', '-dim', type=str, default='umap')

    args = parser.parse_args()
    print(args.inpath)
    bundle = load_trk(args.inpath, "same", bbox_valid_check=False).streamlines
    if len(bundle)>0:
        cleaned_bundle = FiberNeat(bundle, dim_method=args.dim_method)
    else:
        cleaned_bundle = bundle
    save_bundle(cleaned_bundle, args.inpath, args.outpath, verbose=True)

    
    