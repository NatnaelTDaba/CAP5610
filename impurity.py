import numpy as np

def compute_entropy(class_table):

    pit = 1.0*np.array(class_table)/sum(class_table)
    entropy = -sum([p*np.log2(p) if p != 0 else 0 for p in pit])

    return entropy

def compute_children_entropy(class_tables):

    class_tables = np.array(class_tables).T
    n = np.sum(class_tables)
    entropy = sum([(1.0*np.sum(class_tables[:,i])/(n))*(compute_entropy(class_tables[:,i])) for i in range(class_tables.shape[1])])

    return entropy

def compute_gain_split_entropy(parent_table, children_tables):

    gain = compute_entropy(parent_table) - compute_children_entropy(children_tables)

    return gain

def compute_class_error(class_table):

    pit = 1.0*np.array(class_table)/sum(class_table)
    class_error = 1 - np.max(pit)

    return class_error

def compute_children_class_error(class_tables):

    class_tables = np.array(class_tables).T
    n = np.sum(class_tables)
    class_error = sum([(1.0*np.sum(class_tables[:,i])/n)*(compute_class_error(class_tables[:,i])) for i in range(class_tables.shape[1])])

    return class_error

def compute_gain_split_class_error(parent_table, children_tables):

    gain = compute_class_error(parent_table) - compute_children_class_error(children_tables)

    return gain

def compute_gini(class_table):

    pit = 1.0*np.array(class_table)/sum(class_table)
    gini_index = 1-np.sum(pit**2)

    return gini_index

def compute_children_gini(class_tables):

    class_tables = np.array(class_tables).T
    n = np.sum(class_tables)
    gini_index = sum([(1.0*np.sum(class_tables[:,i])/(n))*(compute_gini(class_tables[:,i])) for i in range(class_tables.shape[1])])

    return gini_index

def compute_gain_split_gini(parent_table, children_tables):

    gain = compute_gini(parent_table) - compute_children_gini(children_tables)

    return gain