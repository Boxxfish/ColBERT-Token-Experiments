"""
Helpers for metrics.
"""
from ir_measures import MAP, RR, NDCG

eval_metrics = [MAP(rel=1), MAP(rel=2), MAP(rel=3), RR(rel=1)@10, RR(rel=2)@10, RR(rel=3)@10, NDCG@10, NDCG@1000]