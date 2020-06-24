from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from matplotlib import colors
import scipy.ndimage
from scipy.cluster.vq import vq, kmeans, whiten
from sortedcontainers import SortedSet
import math
import sys
import os
from functools import reduce


class Node:

    def __init__(self, node_id, color_id, pos, size):
        self.node_id = node_id
        self.color_id = color_id
        self.pos = pos
        self.size = size
        self.parent = None
        self.neighbours = set()


    def get_root(self):
        if self.parent is None:
            return self
        else:
            # cache de result:
            self.parent = self.parent.get_root()
            return self.parent


    def merge_in(self, node):
        node.parent = self
        self.size += node.size
        self.neighbours = self.neighbours.union(node.neighbours)
        self.neighbours = set([x.get_root() for x in self.neighbours if x.get_root() != self])


def sample(image_1d, sampling_rate):
    sample_size = int(image_1d.shape[0] * sampling_rate)
    print("Choosing", sample_size, "pixels u.a.r")
    l = []
    for i in range(sample_size):
        l.append(image_1d[int(np.random.random() * image_1d.shape[0])])
    return np.asarray(l)
    image_1d = np.asarray(image_1d)


def to_graph(colors_id):
    n, m = colors_id.shape
    nodes_id = np.zeros((n, m))
    graph = {}
    last_id = 1

    print("Creating nodes")
    for i in range(n):
        for j in range(m):
            if nodes_id[i, j] == 0:
                node_id = last_id
                print("Finding pixels for node:", node_id, " // ", ((i * m + j) * 100) // (n*m), "%")
                color_id = colors_id[i, j]
                nodes_id[i, j] = node_id
                size = 0
                q = [(i, j)]
                while len(q) != 0:
                    a, b = q.pop(0)
                    size += 1
                    for x in range(a-1, a+2):
                        for y in range(b-1, b+2):
                            if (a == x or b == y) and 0 <= x < n and 0 <= y < m:
                                if nodes_id[x, y] == 0 and colors_id[x, y] == color_id:
                                    nodes_id[x, y] = node_id
                                    q.append((x, y))
                graph[node_id] = Node(node_id, color_id, (i, j), size)
                last_id += 1

    visited = np.zeros((n, m))
    print("Creating edges")
    for i in range(n):
        for j in range(m):
            if visited[i, j] == 0:
                node_id = nodes_id[i, j]
                print("Finding edges for node:", node_id, " // ", ((i * m + j) * 100) // (n*m), "%")
                q = [(i, j)]
                while len(q) != 0:
                    a, b = q.pop(0)
                    for x in range(a-1, a+2):
                        for y in range(b-1, b+2):
                            if (a == x or b == y) and 0 <= x < n and 0 <= y < m:
                                if nodes_id[x, y] == node_id and visited[x, y] == 0:
                                    visited[x, y] = 1
                                    q.append((x, y))
                                elif nodes_id[x, y] != node_id:
                                    graph[node_id].neighbours.add(graph[nodes_id[x, y]])
    return nodes_id, graph


def collapse_nodes(graph, threshold=1000):
    surviving_nodes = SortedSet(graph.values(), lambda node: node.size)
    # This can be improved by using a custom heap like in LotterySampling
    min_node = surviving_nodes.pop(0)
    while min_node.size < threshold:
        print("Surviving nodes:", len(surviving_nodes), "-- Min size:", min_node.size)
        assert min_node.parent is None
        max_neighbour = None
        for node in min_node.neighbours:
            root_node = node.get_root()
            assert root_node in surviving_nodes
            if max_neighbour is None or max_neighbour.size < root_node.size:
                max_neighbour = root_node
        surviving_nodes.remove(max_neighbour)
        max_neighbour.merge_in(min_node)
        surviving_nodes.add(max_neighbour)
        min_node = surviving_nodes.pop(0)


def demo(path, sampling_rate=1.0, k=10, threshold=500):
    os.mkdir(path + "_results")
    fig = plt.figure()
    fig.canvas.manager.full_screen_toggle()
    fig.show()
    plt.ion()
    plt.show()

    print(path)
    image = imread(path).astype(float) / 255
    # image = image[:1000, :1000, :]

    (n, m, _) = image.shape
    print(image.shape)
    
    plt.imshow(image, vmax=1)
    plt.draw()
    plt.pause(0.01)
    input("Press [enter] to continue.")


    image_1d = image.reshape(-1, 3)
    if sampling_rate != 1.0:
        train_data = sample(image_1d, sampling_rate)
    else:
        train_data = image_1d
        
    print("Running k-means")
    codebook, _ = kmeans(train_data, k)
    print("Clusters:")
    print(codebook)
    input("Press [enter] to continue.")


    print("Result raw clustering:")
    colors_id, _ = vq(image_1d, codebook)
    colors_id = colors_id.reshape(n, m)
    for i in range(n):
        for j in range(m):
            image[i][j][:] = codebook[colors_id[i][j]]
    plt.imshow(image, vmax=1)
    plt.draw()
    plt.pause(0.01)
    plt.savefig(path + "_results/1_kmeans_raw_clustering.png", quality=100)
    input("Press [enter] to continue.")


    print("Aggrouping pixels into nodes")
    nodes_id, graph = to_graph(colors_id)
    plt.imshow(nodes_id, vmax=len(graph), cmap=colors.ListedColormap(np.random.rand(len(graph), 3)))
    plt.draw()
    plt.pause(0.01)
    plt.savefig(path + "_results/2_nodes.png", quality=100)
    input("Press [enter] to continue.")

    print("Collapsing nodes:")
    collapse_nodes(graph, threshold)
    for i in range(n):
        for j in range(m):
            nodes_id[i][j] = graph[nodes_id[i, j]].get_root().node_id
    plt.imshow(nodes_id, vmax=len(graph), cmap=colors.ListedColormap(np.random.rand(len(graph), 3)))
    plt.draw()
    plt.pause(0.01)
    plt.savefig(path + "_results/3_collapsed_nodes.png", quality=100)
    input("Press [enter] to continue.")


    print("Generating final image:")
    for i in range(n):
        for j in range(m):
            image[i][j][:] = codebook[graph[nodes_id[i, j]].get_root().color_id]
    plt.imshow(image, vmax=1)
    plt.draw()
    plt.pause(0.01)
    plt.savefig(path + "_results/4_final.png", quality=100)
    input("Press [enter] to continue.")


def main():
    np.random.seed(1)

    path = sys.argv[1]
    sampling_rate = float(sys.argv[2])
    k = int(sys.argv[3])
    threshold = int(sys.argv[4])

    demo(path, sampling_rate, k, threshold)

main()