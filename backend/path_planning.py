import networkx as nx
import geopy.distance
import numpy as np
import pandas as pd
from itertools import combinations, permutations

def shortest_walk(a,G):
    '''
    Takes in input the starting node a,  a nx.Graph object and computes
    the shortest walk passing for all nodes.
    '''
    list_nodes = list(G.nodes)
    path = []
    path.append(a)
    while(len(path) != len(list_nodes)):
        #the new node from which to extend the path is the last one inserted in the path
        node = path[-1]
        adj = [x for x in list(G.adj[node]) if x not in path]
        w_list = []
        #I select the neighbor with minimum distance
        for ad in adj:
            w_list.append(G[node][ad]['weight'])
        next_node = adj[np.argmin(w_list)]
        path.append(next_node)
    return path
    
    
def path_planning(clust, centers, DIR_PATH):
    '''
    Takes as input a dictionary that has the various clusters as keys 
    and the nodes associated with each cluster as values and the path 
    of the database. Returns a dict containing the paths of bins ordered
    and a value 'trucks' that indicates the trucks ids associated to 
    each cluster.
    '''
    trucks_assign = truck_cluster_assignments(DIR_PATH, centers)

    df = pd.read_csv(DIR_PATH + '/DATABASE/coords_groups.csv')
    result = {}
    #for each cluster of nodes I calculate the distances using the coordinates
    for cl, tr in zip(clust, trucks_assign.keys()):
        d = [tr]
        d = d + clust[cl]
        diz_edges = {}
        for i in range(len(d)):

            if i!=0:
                coords_1 = df.loc[df["id"] == d[i],['latitude','longitude']].values
            else:
                coords_1 = trucks_assign[tr]

            if i+1 > len(d):
                break
            else:
                for j in range(i+1,len(d)):
                    coords_2 = df.loc[df["id"] == d[j],['latitude','longitude']].values
                    dist = geopy.distance.distance(coords_1, coords_2).m                
                    diz_edges[(d[i],d[j])] = dist
        #I create a complete graph where each edge has as weights the distance as the crow flies between the coordinates associated with the nodes
        G = nx.Graph((x, y, {'weight':v}) for (x,y), v in diz_edges.items())
        #I apply the planning algorithm and return the ordered dictionary
        path = shortest_walk(d[0],G)
        result['zone_'+str(cl)] = path
    
    trucks_ids = []
    for key in result:
        trucks_ids.append(result[key].pop(0))

    result['trucks'] = trucks_ids
    return result

def truck_cluster_assignments(DIR_PATH, centers):

    '''
    Function that takes accesses the DB and assignes the available trucks 
    to the relative cluster by measuring the distance between each truck 
    and the cluster centers. The assignments is performed such that the sum
    of distances between the trucks and centers is minimum.
    
    '''
    trucks_coords =  pd.read_csv(DIR_PATH + '/DATABASE/trucks_coords.csv')
    
    truck_ids = trucks_coords.loc[trucks_coords['available']==1, 'truck_id'].values
    trucks_coords = trucks_coords.loc[ trucks_coords['available']==1, ['latitude', 'longitude']].values

    dists = []

    for c in centers:
        temp = []
        for t in trucks_coords:
            temp.append(geopy.distance.distance(c, t).m)
        dists.append(temp)

    dists = np.array(dists)
    t = [i for i in range(len(truck_ids))]

    combs = list(combinations(t, len(centers) ))
    perms = []
    for c in combs:
        perms = perms + list(permutations(c))

    min_vals = []
    for val in perms:
        min_sum = 0
        for i in range(len(val)):
            min_sum += dists[i, val[i]]
        min_vals.append(min_sum)
    assigned_trucks = { truck_ids[t]:trucks_coords[t] for t in perms[np.argmin(min_vals)] }

    return assigned_trucks
    
    
    
