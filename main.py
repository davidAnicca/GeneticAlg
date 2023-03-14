import genetics
import modules.get_from_gml as get_from_file
import networkx as nx
import matplotlib.pyplot as plt

G = get_from_file.get_graph("./proteins/4.gml")

coms = genetics.genetic_community_detection(G)
print(coms)

grp = []

for community in coms:
    grp.append(list(community))

n_grp = len(grp)

clr = ["red", "blue", "orange", "green", "yellow", "gray", "pink", "purple"]

clr_map = []

for node in G:
    for i in range(n_grp):
        if node in grp[i]:
            clr_map.append(clr[i % len(clr)])

nx.draw(G, node_color=clr_map, with_labels=True)
print("Nr de comunitati: ", n_grp)
plt.show()
