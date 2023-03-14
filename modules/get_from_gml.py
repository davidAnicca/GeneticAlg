import networkx as gf
import matplotlib.pyplot as plt


def get_graph(path):
    G = gf.read_gml(path, label='id')
    return G
