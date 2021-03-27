from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from collections import defaultdict

def kmeans_subdivision(bin_ids, DIR_PATH, k):

    '''
    Function that takes the ids of the bins and splits them into 
    zones by performing a KMeans clustering. Takes in input the array of
    bin ids, accesses the DB and takes coordinates and performs the 
    clustering. Returns a dictionary of the form {zone_id:[bin_id1, bin_id2,...], ...},
    and the cluster's centroids.
    '''

    data = pd.read_csv(DIR_PATH + '/DATABASE/coords_groups.csv')
    coords = data[['latitude','longitude']].values
    coords = coords[bin_ids]
    mean = np.mean(coords, axis=0)
    std = np.std(coords, axis=0)
    coords = (coords - mean)/std

    cl = KMeans(k)
    cl.fit(coords)

    centroids = cl.cluster_centers_*std + mean

    out = defaultdict(list)
    for key, val in zip(cl.labels_, bin_ids):
        out[key].append(val)

    return out, centroids
