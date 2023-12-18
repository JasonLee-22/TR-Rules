import numpy as np


def score1(rule, c=3, weight = 1.0, weight_constraint = 0.5, weight_no_constraint = 0.5):



    #score = rule["rule_supp"] / (rule["body_supp"] + c)
    if not rule["acyclic"]:
        score = weight * rule["rule_supp"] / (rule["body_supp"] + c)
    else:
        if rule["no_constraint"]:
            score = weight_no_constraint * rule["rule_supp"] / (rule["body_supp"] + c)
        else:
            score = weight_constraint * rule["rule_supp"] / (rule["body_supp"] + c)

    return score


def score2(cands_walks, test_query_ts, lmbda):


    max_cands_ts = max(cands_walks["timestamp_0"])
    score = np.exp(
        lmbda * (max_cands_ts - test_query_ts)
    )  # Score depending on time difference

    return score


def score_12(rule, cands_walks, test_query_ts, lmbda, a):
    score = a * score1(rule) + (1 - a) * score2(cands_walks, test_query_ts, lmbda)

    return score
