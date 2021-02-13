import numpy as np

def compute_entropy(class_table):

    pit = 1.0*np.array(class_table)/sum(class_table)
    entropy = -sum([p*np.log2(p) if p != 0 else 0 for p in pit])

    return entropy

def compute_children_entropy(class_tables):

    class_tables = np.array(class_tables)
    n = np.sum(class_tables)
    entropy = sum([(1.0*np.sum(class_tables[:,i])/(n*1.0))*(compute_entropy(class_tables[:,i])) for i in range(len(class_tables))])

    return entropy

def compute_gain_split(parent_table, children_tables):

    gain = compute_entropy(parent_table) - compute_children_entropy(children_tables)

    return gain