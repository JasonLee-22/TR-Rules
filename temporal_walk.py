import numpy as np
import scipy.sparse as sp

class Temporal_Walk(object):
    def __init__(self, learn_data, inv_relation_id, transition_distr, dataset_dir, train_times):

        self.dataset_dir = dataset_dir
        self.learn_data = learn_data
        self.inv_relation_id = inv_relation_id
        self.transition_distr = transition_distr
        self.neighbors = store_neighbors(learn_data)
        self.edges = store_edges(learn_data)
        self.num_r = len(inv_relation_id)
        self.num_time = train_times

    def sample_start_edge(self, rel_idx, s=1):
        rel_edges = self.edges[rel_idx]
        start_edge = rel_edges[np.random.choice(len(rel_edges))]
        return start_edge

    def sample_next_edge(self, filtered_edges, cur_ts, s=1):
        if self.transition_distr == "unif":
            next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
        elif self.transition_distr == "exp":
            tss = filtered_edges[:, 3]
            prob = np.exp(tss - cur_ts)
            try:
                prob = prob / np.sum(prob)
                next_edge = filtered_edges[
                    np.random.choice(range(len(filtered_edges)), p=prob)
                ]
            except ValueError:  # All timestamps are far away
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
        return next_edge

    def transition_step(self, cur_node, cur_ts, prev_edge, start_node, step, L, freq_mat):

        next_edges = self.neighbors[cur_node]

        if step == 1:  # The next timestamp should be smaller than the current timestamp
            filtered_edges = next_edges[next_edges[:, 3] < cur_ts]
        else:  # The next timestamp should be smaller than or equal to the current timestamp
            filtered_edges = next_edges[next_edges[:, 3] <= cur_ts]
            # Delete inverse edge
            inv_edge = [
                cur_node,
                self.inv_relation_id[prev_edge[1]],
                prev_edge[0],
                cur_ts,
            ]
            row_idx = np.where(np.all(filtered_edges == inv_edge, axis=1))
            filtered_edges = np.delete(filtered_edges, row_idx, axis=0)

        if step == L - 1:  # Find an edge that connects to the source of the walk
            filtered_edges = filtered_edges[filtered_edges[:, 2] == start_node]

        if len(filtered_edges):
            next_edge = self.sample_next_edge(filtered_edges, cur_ts, freq_mat)
        else:
            next_edge = []

        return next_edge

    def sample_walk(self, L, rel_idx):

        walk_successful = True
        walk = dict()
        prev_edge = self.sample_start_edge(rel_idx)
        start_node = prev_edge[0]
        cur_node = prev_edge[2]
        cur_ts = prev_edge[3]
        walk["entities"] = [start_node, cur_node]
        walk["relations"] = [prev_edge[1]]
        walk["timestamps"] = [cur_ts]

        for step in range(1, L):
            next_edge = self.transition_step(
                cur_node, cur_ts, prev_edge, start_node, step, L, freq_mat
            )
            if len(next_edge):
                cur_node = next_edge[2]
                cur_ts = next_edge[3]
                walk["relations"].append(next_edge[1])
                walk["entities"].append(cur_node)
                walk["timestamps"].append(cur_ts)
                prev_edge = next_edge
            else:  # No valid neighbors (due to temporal or cyclic constraints)
                walk_successful = False
                break

        return walk_successful, walk

    def Acyclic_sample(self, rel, s=1):
        walk_successful = True
        walk = dict()
        prev_edge = self.sample_start_edge(rel)
        start_node = prev_edge[0]
        cur_node = prev_edge[2]
        cur_ts = prev_edge[3]
        walk["entities"] = [start_node, cur_node]
        walk["relations"] = [prev_edge[1]]
        walk["timestamps"] = [cur_ts]

        next_edges = self.neighbors[start_node]
        #mask = (next_edges[:, 3] < cur_ts) * (next_edges[:, 2] != cur_node)
        mask = (next_edges[:, 3] < cur_ts)
        filtered_edges = next_edges[mask]
        if len(filtered_edges):
            if self.transition_distr == 'unif':
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
            elif self.transition_distr == 'exp':
                tss = filtered_edges[:, 3]
                prob = np.exp(tss - cur_ts)
                try:
                    prob = prob / np.sum(prob)
                    next_edge = filtered_edges[
                        np.random.choice(range(len(filtered_edges)), p=prob)
                    ]
                except ValueError:  # All timestamps are far away
                    next_edge = filtered_edges[np.random.choice(len(filtered_edges))]

            walk["relations"].append(next_edge[1])
            walk["entities"].append(next_edge[2])
            walk["timestamps"].append(next_edge[3])
        else:
            walk_successful = False

        return walk_successful, walk
        #For X r a, we find X r' b that indicates the former stands.
    #Body supp = how many X r' b  Rule supp = how many make X r a stand

def store_neighbors(quads):

    neighbors = dict()
    nodes = list(set(quads[:, 0]))
    for node in nodes:
        neighbors[node] = quads[quads[:, 0] == node]

    return neighbors


def store_edges(quads):

    edges = dict()
    relations = list(set(quads[:, 1]))
    for rel in relations:
        edges[rel] = quads[quads[:, 1] == rel]

    return edges
