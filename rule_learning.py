import os
import json
import itertools
import numpy as np
from collections import Counter
import operator


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class Rule_Learner(object):
    def __init__(self, edges, id2relation, inv_relation_id, dataset):
        self.edges = edges
        self.id2relation = id2relation
        self.inv_relation_id = inv_relation_id

        self.found_rules = []
        self.rules_dict = dict()
        self.output_dir = "../output/" + dataset + "/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def create_rule(self, walk):

        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            self.inv_relation_id[x] for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"] = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )
        rule["acyclic"] = False

        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())
            (
                rule["conf"],
                rule["rule_supp"],
                rule["body_supp"],
            ) = self.estimate_confidence(rule)

            if rule["conf"]:
                self.update_rules_dict(rule)

    def create_acyclic(self, walk):

        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            int(walk["relations"][-1])
        ]
        rule["var_constraints"] = []
        rule["acyclic"] = True
        rule["no_constraint"] = False
        rule["constraint_tail"] = int(walk["entities"][-1])
        rule["head_constraint"] = int(walk["entities"][1])
        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())
            (
                rule["conf"],
                rule["rule_supp"],
                rule["body_supp"],
            ) = self.estimate_acyclic_confidence(rule)

            if rule["conf"]:
                self.update_rules_dict(rule)


    def define_var_constraints(self, entities):
        var_constraints = []
        for ent in set(entities):            #[x,y,x,z] (x,y,z) --> [[0, 2], [1]. [3]]
            all_idx = [idx for idx, x in enumerate(entities) if x == ent]
            var_constraints.append(all_idx)
        var_constraints = [x for x in var_constraints if len(x) > 1]

        return sorted(var_constraints)

    def estimate_confidence(self, rule, num_samples=500, weight=1.0, s=3):
        all_bodies = []
        for _ in range(num_samples):
            sample_successful, body_ents_tss = self.sample_body(
                rule["body_rels"], rule["var_constraints"]
            )
            if sample_successful:
                all_bodies.append(body_ents_tss)

        all_bodies.sort()
        unique_bodies = list(x for x, _ in itertools.groupby(all_bodies))
        body_support = len(unique_bodies)

        confidence, rule_support = 0, 0
        if body_support:
            rule_support = self.calculate_rule_support(unique_bodies, rule["head_rel"])
            confidence = rule_support / (body_support + s)
            confidence = round(confidence * weight, 6)

        return confidence, rule_support, body_support
    #  X r b -> X r' a
    #
    def estimate_acyclic_confidence(self, rule, num_samples=500, weight_constraint=1.0, s=3):
        all_bodies = []
        for _ in range(num_samples):
            sample_successful, body_ents_tss = self.sample_acyclic_body(
                rule["body_rels"],
                rule["constraint_tail"]
            )
            if sample_successful:
                all_bodies.append(body_ents_tss)

        all_bodies.sort()
        unique_bodies = list(x for x, _ in itertools.groupby(all_bodies))
        body_support = len(unique_bodies)

        confidence, rule_support = 0, 0
        if body_support:
            rule_support = self.calculate_acyclic_rule_support(unique_bodies, rule["head_rel"], rule["head_constraint"])
            confidence = rule_support / (body_support + s)
            confidence = round(confidence * weight_constraint, 6)
        return confidence, rule_support, body_support

    def sample_body(self, body_rels, var_constraints):
        sample_successful = True
        body_ents_tss = []
        cur_rel = body_rels[0]
        rel_edges = self.edges[cur_rel]
        next_edge = rel_edges[np.random.choice(len(rel_edges))]
        cur_ts = next_edge[3]
        cur_node = next_edge[2]
        body_ents_tss.append(next_edge[0])
        body_ents_tss.append(cur_ts)
        body_ents_tss.append(cur_node)

        for cur_rel in body_rels[1:]:
            next_edges = self.edges[cur_rel]
            mask = (next_edges[:, 0] == cur_node) * (next_edges[:, 3] >= cur_ts)
            filtered_edges = next_edges[mask]

            if len(filtered_edges):
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
                cur_ts = next_edge[3]
                cur_node = next_edge[2]
                body_ents_tss.append(cur_ts)
                body_ents_tss.append(cur_node)
            else:
                sample_successful = False
                break

        if sample_successful and var_constraints:
            # Check variable constraints
            body_var_constraints = self.define_var_constraints(body_ents_tss[::2])
            if body_var_constraints != var_constraints:
                sample_successful = False

        return sample_successful, body_ents_tss

    def sample_acyclic_body(self, body_rels, constraint_tail):
        """
        Sample an acyclic walk according to the rule body.
        """
        sample_successful = True
        body_ents_tss = []
        cur_rel = body_rels[0]
        rel_edges = self.edges[cur_rel]
        mask = rel_edges[:,2] == constraint_tail
        rel_edges = rel_edges[mask]
        if len(rel_edges):
            next_edge = rel_edges[np.random.choice(len(rel_edges))]
            cur_ts = next_edge[3]
            cur_node = next_edge[2]
            body_ents_tss.append(next_edge[0])
            body_ents_tss.append(cur_ts)
            body_ents_tss.append(cur_node)
        else:
            sample_successful = False

        return sample_successful, body_ents_tss


    def calculate_rule_support(self, unique_bodies, head_rel):

        rule_support = 0
        head_rel_edges = self.edges[head_rel]
        before_body = [-1, -1, -1]
        before_mask = np.array([-1, -1, -1])
        unique_bodies.sort(key=lambda x: x[::2] + [x[-2]])
        for body in unique_bodies:
            mask = (
                (head_rel_edges[:, 0] == body[0])
                * (head_rel_edges[:, 2] == body[-1])
                * (head_rel_edges[:, 3] > body[-2])
            )


            if before_body == [-1, -1, -1] and True in mask:
                rule_support += 1
                before_body = body
                before_mask = mask
                continue

            #print(operator.ne(mask.tolist(), before_mask.tolist()))
            if before_body[::2] == body[::2] and operator.ne(mask.tolist(), before_mask.tolist()) and True in mask:
                #print('2 fired')
                rule_support += 1
                before_body = body
                before_mask = mask
                continue

            if before_body[::2] != body[::2] and True in mask:
                rule_support += 1
                before_body = body
                before_mask = mask
                continue

            before_body = body
            before_mask = mask

        return rule_support

    def calculate_acyclic_rule_support(self, unique_bodies, head_rel, head_constraint):

        rule_support = 0
        head_rel_edges = self.edges[head_rel]
        before_body = [-1, -1, -1]
        before_mask = np.array([-1, -1, -1])
        unique_bodies.sort(key=lambda x: [x[0], x[-1], x[-2]])
        #print(unique_bodies)
        for body in unique_bodies:
            mask = (
                (head_rel_edges[:, 0] == body[0])
                * (head_rel_edges[:, 3] > body[-2])
                * (head_rel_edges[:, 2]== head_constraint)
            )

            if True in mask:
                rule_support += 1

            if before_body == [-1, -1, -1] and True in mask:
                rule_support += 1
                before_body = body
                before_mask = mask
                continue

            #print(operator.ne(mask.tolist(), before_mask.tolist()))
            if before_body[::2] == body[::2] and operator.ne(mask.tolist(), before_mask.tolist()) and True in mask:
                #print('2 fired')
                rule_support += 1
                before_body = body
                before_mask = mask
                continue

            if before_body[::2] != body[::2] and True in mask:
                rule_support += 1
                before_body = body
                before_mask = mask
                continue

            before_body = body
            before_mask = mask
        return rule_support

    def update_rules_dict(self, rule):

        try:
            self.rules_dict[rule["head_rel"]].append(rule)
        except KeyError:
            self.rules_dict[rule["head_rel"]] = [rule]

    def sort_rules_dict(self):

        for rel in self.rules_dict:
            self.rules_dict[rel] = sorted(
                self.rules_dict[rel], key=lambda x: x["conf"], reverse=True
            )

    def save_rules(self, dt, rule_lengths, num_walks, transition_distr, seed):


        rules_dict = {int(k): v for k, v in self.rules_dict.items()}
        filename = "{0}_r{1}_n{2}_{3}_s{4}_rules.json".format(
            dt, rule_lengths, num_walks, transition_distr, seed
        )
        filename = filename.replace(" ", "")
        with open(self.output_dir + filename, "w", encoding="utf-8") as fout:
            json.dump(rules_dict, fout)

    def save_rules_verbalized(
        self, dt, rule_lengths, num_walks, transition_distr, seed
    ):

        rules_str = ""
        for rel in self.rules_dict:
            for rule in self.rules_dict[rel]:
                rules_str += verbalize_rule(rule, self.id2relation) + "\n"

        filename = "{0}_r{1}_n{2}_{3}_s{4}_rules.txt".format(
            dt, rule_lengths, num_walks, transition_distr, seed
        )
        filename = filename.replace(" ", "")
        with open(self.output_dir + filename, "w", encoding="utf-8") as fout:
            fout.write(rules_str)


def verbalize_rule(rule, id2relation):

    if rule["var_constraints"]:
        var_constraints = rule["var_constraints"]
        constraints = [x for sublist in var_constraints for x in sublist]
        for i in range(len(rule["body_rels"]) + 1):
            if i not in constraints:
                var_constraints.append([i])
        var_constraints = sorted(var_constraints)
    else:
        var_constraints = [[x] for x in range(len(rule["body_rels"]) + 1)]

    rule_str = "{0:8.6f}  {1:4}  {2:4}  {3}(X0,X{4},T{5}) <- "
    obj_idx = [
        idx
        for idx in range(len(var_constraints))
        if len(rule["body_rels"]) in var_constraints[idx]
    ][0]
    rule_str = rule_str.format(
        rule["conf"],
        rule["rule_supp"],
        rule["body_supp"],
        id2relation[rule["head_rel"]],
        obj_idx,
        len(rule["body_rels"]),
    )

    for i in range(len(rule["body_rels"])):
        sub_idx = [
            idx for idx in range(len(var_constraints)) if i in var_constraints[idx]
        ][0]
        obj_idx = [
            idx for idx in range(len(var_constraints)) if i + 1 in var_constraints[idx]
        ][0]
        rule_str += "{0}(X{1},X{2},T{3}), ".format(
            id2relation[rule["body_rels"][i]], sub_idx, obj_idx, i
        )

    return rule_str[:-2]


def rules_statistics(rules_dict):

    print(
        "Number of relations with rules: ", len(rules_dict)
    )  # Including inverse relations
    print("Total number of rules: ", sum([len(v) for k, v in rules_dict.items()]))

    lengths = []
    for rel in rules_dict:
        lengths += [len(x["body_rels"]) for x in rules_dict[rel]]
    rule_lengths = [(k, v) for k, v in Counter(lengths).items()]
    print("Number of rules by length: ", sorted(rule_lengths))
