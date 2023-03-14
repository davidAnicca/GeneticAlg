import networkx as gr
import matplotlib.pyplot as plt

edges = ([2, 3],
        [2, 4],
        [4, 5],
        [3, 5])

G = gr.Graph(edges)

subax1 = plt.subplot(121)
gr.draw(G, with_labels=True, font_weight='bold')
#subax2 = plt.subplot(122)
#gr.draw_shell(G, nlist=[range(5, 10), range(9)], with_labels=True, font_weight='bold')

plt.show()

