import metis
from utils import *

def partition_graph(adj, idx_nodes, num_clusters):
    """partition a graph by METIS."""

    start_time = time.time()
    num_nodes = len(idx_nodes)

    train_adj_lil = adj[idx_nodes, :][:, idx_nodes].tolil()
    train_adj_lists = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        rows = train_adj_lil[i].rows[0]
        if i in rows:
            rows.remove(i)
        train_adj_lists[i] = rows

    if num_clusters > 1:
        _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    else:
        groups = [0] * num_nodes

    parts = [[] for _ in range(num_clusters)]
    for nd_idx in range(num_nodes):
        gp_idx = groups[nd_idx]
        parts[gp_idx].append(idx_nodes[nd_idx])
    
    print("Partitioning done. %f seconds." % (time.time() - start_time))
    return parts
