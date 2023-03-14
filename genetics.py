import networkx as nx
import numpy as np
import pandas as pd


def genetic_community_detection(network, population=500, generations=50, alpha=3.45, mutation_rate=0.2):
    a_matrix = nx.adjacency_matrix(network)
    nodes_count = len(network.nodes())

    d = {"chromosome": [generate_chromosome(nodes_count, a_matrix) for n in range(population)]}
    data_frame = pd.DataFrame(data=d)
    data_frame["subsets"] = data_frame["chromosome"].apply(find_subsets)
    data_frame["Q"] = data_frame.apply(lambda x: community_score_q(x["chromosome"], x["subsets"], alpha, a_matrix),
                                       axis=1)

    current_gen = 0
    population_count = population
    while current_gen < generations:
        print("generatia: ", current_gen)
        data_frame.to_csv("./data/" + str(current_gen)+".csv")
        for i in range(int(np.floor(population / 10))):

            # ordonare în funcție de Q (identificarea elitelor)
            elites = data_frame.sort_values("Q", ascending=True)[int(np.floor(population / 10)):]

            # identificarea a doi părinți folosind metoda selecției
            mom = selection(elites)
            dad = selection(elites)

            # copilul se produce făcând crossover la informațiile părinților
            child = crossover(data_frame["chromosome"][mom], data_frame["chromosome"][dad], 0.8)

            # ignorare copii cu dizabilități :)
            if len(child) == 0:
                continue

            # mutatie asupra copilului
            child = mutation(child, a_matrix, mutation_rate)

            # submultimi (comunități provizorii)
            child_subsets = find_subsets(child)

            # calcularea scorului
            child_cs = community_score_q(child, child_subsets, alpha, a_matrix)

            # introducere copil în populatie
            data_frame.loc[population_count] = [child, child_subsets, child_cs]
            population_count += 1

        data_frame_sorted_des = data_frame.sort_values("Q", ascending=False)

        # păstrare în populație cei cu Q mare (de regulă noii copii băgați)
        to_drop = data_frame_sorted_des.index[population:]
        data_frame.drop(to_drop, inplace=True)

        # trecere la următoarea generație
        current_gen += 1

    des_data_frame = data_frame.sort_values("Q", ascending=False).index[0]

    nodes_subsets = data_frame["subsets"][des_data_frame]
    nodes_list = list(network.nodes())
    result = []
    for subs in nodes_subsets:
        subset = []
        for n in subs:
            subset.append(nodes_list[n])
        result.append(subset)
    return result


def generate_chromosome(nodes_length, Adj):
    chromosome = np.array([], dtype=int)
    for x in range(nodes_length):
        rand = np.random.randint(0, nodes_length)
        while Adj[x, rand] != 1:
            rand = np.random.randint(0, nodes_length)
        chromosome = np.append(chromosome, rand)
    return chromosome


def merge_subsets(sub):
    arr = []
    to_skip = []
    for s in range(len(sub)):
        if sub[s] not in to_skip:
            new = sub[s]
            for x in sub:
                if sub[s] & x:
                    new = new | x
                    to_skip.append(x)
            arr.append(new)
    return arr


def find_subsets(chromosome):
    sub = [{x, chromosome[x]} for x in range(len(chromosome))]
    result = sub
    i = 0
    while i < len(result):
        candidate = merge_subsets(result)
        if candidate != result:
            result = candidate
        else:
            break
        result = candidate
        i += 1
    return result


# community score of a clustering (p19)
def community_score_q(chrom, subsets, alpha, Adj):
    matrix = Adj.toarray()
    score = 0
    for s in subsets:
        small_matrix = np.zeros((len(chrom), len(chrom)), dtype=int)
        for i in s:
            for j in s:
                small_matrix[i][j] = matrix[i][j]
        m = 0
        v = 0
        for row in list(s):
            row_mean = np.sum(small_matrix[row]) / len(s)
            v += np.sum(small_matrix[row])
            m += (row_mean ** alpha) / len(s)
        score += m * v
    return score


# roulette selection based on score
def selection(df_elites):
    random_proportion = np.random.random_sample()
    sum_cs = np.sum(df_elites["Q"])
    q = 0
    for i in df_elites.index:
        q += df_elites["Q"][i]
        proportion = q / sum_cs
        if random_proportion < proportion:
            chosen = i
            break
    return chosen


def crossover(parent_1, parent_2, crossover_rate):
    if np.random.random_sample() < crossover_rate:
        length = len(parent_1)
        mask = np.random.randint(2, size=length)
        child = np.zeros(length, dtype=int)
        for i in range(len(mask)):
            if mask[i] == 1:
                child[i] = parent_1[i]
            else:
                child[i] = parent_2[i]
        return child
    else:
        return np.array([])


def mutation(chrom, Adj, mutation_rate):
    if np.random.random_sample() < mutation_rate:
        chrom = chrom
        neighbor = []
        while len(neighbor) < 2:
            mutant = np.random.randint(1, len(chrom))
            matr = Adj.toarray()
            row = matr[mutant]
            neighbor = [i for i in range(len(row)) if row[i] == 1]
            if len(neighbor) > 1:
                neighbor.remove(chrom[mutant])
                to_change = int(np.floor(np.random.random_sample() * (len(neighbor))))
                chrom[mutant] = neighbor[to_change]
                neighbor.append(chrom[mutant])
    return chrom
