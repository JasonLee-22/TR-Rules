import json
import numpy as np
import pandas as pd
from collections import Counter
from temporal_walk import store_edges

def filter_rules(rules_dict, min_conf, min_body_supp, rule_lengths, acyclic_min=10):
    new_rules_dict = dict()
    for k in rules_dict:
        new_rules_dict[k] = []
        for rule in rules_dict[k]:
            cond = (
                    (rule["conf"] >= min_conf)
                    and (rule["body_supp"] >= min_body_supp)
                    and (len(rule["body_rels"]) in rule_lengths)
            )

            if cond:
                new_rules_dict[k].append(rule)

    return new_rules_dict


def get_window_edges(all_data, test_query_ts, learn_edges, window=-1):
    if window > 0:
        mask = (all_data[:, 3] < test_query_ts) * (
            all_data[:, 3] >= test_query_ts - window
        )
        window_edges = store_edges(all_data[mask])
    elif window == 0:
        mask = all_data[:, 3] < test_query_ts
        window_edges = store_edges(all_data[mask])
    elif window == -1:
        window_edges = learn_edges

    return window_edges


def match_body_relations(rule, edges, test_query_sub):

    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_sub
        new_edges = rel_edges[mask]
        walk_edges = [
            np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))
        ]  # [sub, obj, ts]
        cur_targets = np.array(list(set(walk_edges[0][:, 1])))

        for i in range(1, len(rels)):
            # Match current targets and next body relation
            try:
                rel_edges = edges[rels[i]]
                mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
                new_edges = rel_edges[mask]
                walk_edges.append(
                    np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))
                )  # [sub, obj, ts]
                #print('correct edges: ', walk_edges[-1])
                #print('cyclic: ', walk_edges[-1].shape)
                cur_targets = np.array(list(set(walk_edges[i][:, 1])))
            except KeyError:
                walk_edges.append([])
                break
    except KeyError:
        walk_edges = [[]]

    return walk_edges


def match_acyclic_bodies(rule, edges, test_query_sub):

    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]
        if rule["no_constraint"] == False:
            mask = (rel_edges[:, 0] == test_query_sub) * (rel_edges[:, 2] == rule["constraint_tail"])
        elif rule["no_constraint"] == True:
            mask = rel_edges[:, 0] == test_query_sub
        if True in mask:
            max_t = max(rel_edges[mask][:, 3])
            walk_edges = [np.expand_dims(np.hstack((test_query_sub, rule["head_constraint"], max_t)), axis=0)]

        else:
            walk_edges = [[]]
    except KeyError:
        walk_edges = [[]]
    return walk_edges


def match_body_relations_complete(rule, edges, test_query_sub):


    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_sub
        new_edges = rel_edges[mask]
        walk_edges = [new_edges]
        cur_targets = np.array(list(set(walk_edges[0][:, 2])))

        for i in range(1, len(rels)):
            # Match current targets and next body relation
            try:
                rel_edges = edges[rels[i]]
                mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
                new_edges = rel_edges[mask]
                walk_edges.append(new_edges)
                cur_targets = np.array(list(set(walk_edges[i][:, 2])))
            except KeyError:
                walk_edges.append([])
                break
    except KeyError:
        walk_edges = [[]]

    return walk_edges


def get_walks(rule, walk_edges):


    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=["entity_" + str(0), "entity_" + str(1), "timestamp_" + str(0)],
        dtype=np.uint16,
    )  # Change type if necessary for better memory efficiency
    if not rule["var_constraints"]:
        del df["entity_" + str(0)]
    df_edges.append(df)
    df = df[0:0]  # Memory efficiency

    for i in range(1, len(walk_edges)):
        df = pd.DataFrame(
            walk_edges[i],
            columns=["entity_" + str(i), "entity_" + str(i + 1), "timestamp_" + str(i)],
            dtype=np.uint16,
        )  # Change type if necessary
        df_edges.append(df)
        df = df[0:0]

    rule_walks = df_edges[0]
    df_edges[0] = df_edges[0][0:0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_" + str(i)])
        rule_walks = rule_walks[
            rule_walks["timestamp_" + str(i - 1)] <= rule_walks["timestamp_" + str(i)]
        ]
        if not rule["var_constraints"]:
            del rule_walks["entity_" + str(i)]
        df_edges[i] = df_edges[i][0:0]

    for i in range(1, len(rule["body_rels"])):
        del rule_walks["timestamp_" + str(i)]

    return rule_walks


def get_walks_complete(rule, walk_edges):

    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=[
            "entity_" + str(0),
            "relation_" + str(0),
            "entity_" + str(1),
            "timestamp_" + str(0),
        ],
        dtype=np.uint16,
    )  # Change type if necessary for better memory efficiency
    df_edges.append(df)

    for i in range(1, len(walk_edges)):
        df = pd.DataFrame(
            walk_edges[i],
            columns=[
                "entity_" + str(i),
                "relation_" + str(i),
                "entity_" + str(i + 1),
                "timestamp_" + str(i),
            ],
            dtype=np.uint16,
        )  # Change type if necessary
        df_edges.append(df)

    rule_walks = df_edges[0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_" + str(i)])
        rule_walks = rule_walks[
            rule_walks["timestamp_" + str(i - 1)] <= rule_walks["timestamp_" + str(i)]
        ]

    return rule_walks

def check_var_constraints(var_constraints, rule_walks):

    for const in var_constraints:
        for i in range(len(const) - 1):
            rule_walks = rule_walks[
                rule_walks["entity_" + str(const[i])]
                == rule_walks["entity_" + str(const[i + 1])]
            ]

    return rule_walks


def get_candidates(
    rule, rule_walks, test_query_ts, cands_dict, score_func, args, dicts_idx
):

    max_entity = "entity_" + str(len(rule["body_rels"]))
    cands = set(rule_walks[max_entity])
    for cand in cands:
        cands_walks = rule_walks[rule_walks[max_entity] == cand]
        for s in dicts_idx:
            score = score_func(rule, cands_walks, test_query_ts, *args[s]).astype(
                np.float32
            )
            try:
                cands_dict[s][cand].append(score)
            except KeyError:
                cands_dict[s][cand] = [score]
    return cands_dict


def save_candidates(
    rules_file, dir_path, all_candidates, rule_lengths, window, score_func_str
):

    all_candidates = {int(k): v for k, v in all_candidates.items()}
    for k in all_candidates:
        all_candidates[k] = {int(cand): v for cand, v in all_candidates[k].items()}
    filename = "{0}_cands_r{1}_w{2}_{3}.json".format(
        rules_file[:-11], rule_lengths, window, score_func_str
    )
    filename = filename.replace(" ", "")
    with open(dir_path + filename, "w", encoding="utf-8") as fout:
        json.dump(all_candidates, fout)


def verbalize_walk(walk, data):

    l = len(walk) // 3
    walk = walk.values.tolist()

    walk_str = data.id2entity[walk[0]] + "\t"
    for j in range(l):
        walk_str += data.id2relation[walk[3 * j + 1]] + "\t"
        walk_str += data.id2entity[walk[3 * j + 2]] + "\t"
        walk_str += data.id2ts[walk[3 * j + 3]] + "\t"

    return walk_str[:-1]
