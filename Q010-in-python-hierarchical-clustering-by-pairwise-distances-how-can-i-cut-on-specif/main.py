from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage

obj_distances = {
    ('DN1357_i2', 'DN1357_i5'): 1.0,
    ('DN1357_i2', 'DN10172_i1'): 28.0,
    ('DN1357_i2', 'DN1357_i1'): 8.0,
    ('DN1357_i5', 'DN1357_i1'): 2.0,
    ('DN1357_i5', 'DN10172_i1'): 34.0,
    ('DN1357_i1', 'DN10172_i1'): 38.0,
}

keys = [sorted(k) for k in obj_distances.keys()]
values = obj_distances.values()
sorted_keys, distances = zip(*sorted(zip(keys, values)))

Z = linkage(distances)

labels = sorted(set([key[0] for key in sorted_keys] + [sorted_keys[-1][-1]]))
dendro = dendrogram(Z, labels=labels)

members = dendro['ivl']
clusters = cut_tree(Z, height=2)
cluster_ids = [c[0] for c in clusters]

for k in range(max(cluster_ids) + 1):
    print(f"Cluster {k}")
    for i, c in enumerate(cluster_ids):
        if c == k:
            print(f"{members[i]}")

    print('\n')
