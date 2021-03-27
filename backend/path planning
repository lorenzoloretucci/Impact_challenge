import networkx as nx
import geopy.distance
import numpy as np
import pandas as pd

#takes as input the graph G and a node a
def cammino(a,G):
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
    
    
#It takes as input a dictionary that has the various clusters as keys and the nodes associated with each cluster as values and the path of the database
def path_planning(clust,DIR_PATH):
    df = pd.read_csv(DIR_PATH + '/DATABASE/coords_groups.csv')
    result = {}
    #for each cluster of nodes I calculate the distances using the coordinates
    for cl in clust:
        d = clust[cl]
        diz_edges = {}
        for i in range(len(d)):
            coords_1 = df.loc[df["id"] == d[i],['latitude','longitude']].values
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
        path = cammino(d[0],G)
        result[cl] = path
    return result
    
    
    
    
