from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def kmeans_subdivision(bin_ids, DIR_PATH, k):
    '''
    Function that takes the ids of the bins and splits them into 
    zones by performing a KMeans clustering. Takes in input the array of
    bin ids, accesses the DB and takes coordinates and performs the 
    clustering. Returns an array of (bin_idx, cluster_labels).
    '''
    data = pd.read_csv(DIR_PATH + '/DATABASE/coords_groups.csv')
    coords = data[['latitude','longitude']].values
    coords = coords[bin_ids]
    coords = (coords - np.mean(coords, axis=0))/np.std(coords, axis=0)

    cl = KMeans(k)
    cl.fit(coords)

    return np.stack((bin_ids, cl.labels_), axis=1)