import numpy as np

def precision_at_k(recommended, relevant, k):
    if not recommended: return 0.0
    rec_k = set(recommended[:k])
    rel = set(relevant)
    return len(rec_k & rel) / k

def recall_at_k(recommended, relevant, k):
    if not relevant: return 0.0
    rec_k = set(recommended[:k])
    rel = set(relevant)
    return len(rec_k & rel) / len(rel)