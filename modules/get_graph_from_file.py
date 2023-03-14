import networkx as gf
import matplotlib.pyplot as plt

def read_matrix():

    f = open("network.in", "r")

    n = int(f.readline())

    m = []

    line = f.readline()
    while line != "":
        line_in_matrix = []
        values = line.split()
        for value in values:
            line_in_matrix.append(int(value))
        m.append(line_in_matrix)
        line = f.readline()

    f.close()
    return m, n

def get_graph():
    m, n = read_matrix()
    G = gf.Graph()
    for i in range(n):
        for j in range(n):
            if m[i][j] == 1:
                G.add_edge(i, j)
    return G

def debug():
    matr, n = read_matrix()
    for line in matr:
        print(line)

#debug()
#get_graph()


