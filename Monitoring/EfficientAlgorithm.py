import networkx as nx
from networkx.algorithms.flow import *
from collections import deque


def bfs_res(ResGraph, src):
	heights = dict()

	heights[src] = 0

	q = deque([(src, 0)])
	while q:
		u, height = q.popleft()
		height += 1
		for v, attr in ResGraph.succ[u].items():
			if v not in heights and attr['flow'] < attr['capacity']:
				heights[v] = height
				q.append((v, height))

	for node in set(ResGraph.nodes()) - set(heights.keys()):
		heights[node] = float("Inf")
	return heights


def computePartitions(graph, src, trg, verbose=False):
	n = len(graph.nodes())
	R = preflow_push(graph, src, trg, capacity="weight")

	dists_source = bfs_res(R, src)
	dists_target = bfs_res(R, trg)

	d = dict()
	for v in R.nodes():
		d[v] = min(dists_source[v], dists_target[v] + n)

	S = set()
	for v in R.nodes():
		if d[v] < n:
			S.add(v)

	T = set(graph.nodes())-S
	return S, T


def createGAlpha(graph, alfa, target):
	G_alfa = graph.copy()
	for node in G_alfa.nodes():
		if node != target:
			if target in G_alfa[node]:
				cap = G_alfa[node][target]['weight']
				G_alfa.add_edge(node, target, weight=cap+alfa)
			else:
				G_alfa.add_edge(node, target, weight=alfa)
	return G_alfa


def reachable_residual_graph(ResGraph, src):
	rc = set()
	heights = {src: 0}
	q = deque([(src, 0)])
	rc.add(src)
	while q:
		u, height = q.popleft()
		height += 1
		for v, attr in ResGraph.succ[u].items():
			if v not in heights and ResGraph.succ[u][v]['flow'] < ResGraph.succ[u][v]['weight']:
				heights[v] = height
				q.append((v, height))
				rc.add(v)
	return rc


def cutGraph(cutset, source, target):
	cg = nx.DiGraph()
	if (source, target) in cutset:
		c_st = True
	else:
		c_st = False

	for (u, v) in cutset:
		if u != source:
			if v != target or (v == target and not c_st):
				cg.add_edge(u, v, weight=1)
	return cg


def chooseMs(cutset, source, target, verbose=False):
	cut_graph = cutGraph(cutset, source, target)
	m = set()

	for (u, v) in cutset:
		if u == source and v == target:
			m.add(v)
		else:
			if u == source:
				m.add(v)

	cut_graph_edges = cut_graph.edges()

	for (u, v) in list(cut_graph_edges):
		cut_graph.add_edge('s', u, weight=1)
		cut_graph.add_edge(v, 't', weight=1)

	if len(cut_graph.edges()) != 0:
		R = edmonds_karp(cut_graph, 's', 't')
		A = reachable_residual_graph(R, 's')
		B = set(cut_graph.nodes()) - A
		cut_graph_cutset = computeCutset(cut_graph, A, B)

		for (u, v) in cut_graph_cutset:
			if u == 's':
				m.add(v)
			else:
				m.add(u)
	return m


def computeCutset(Gr, S, T):
	c = set()
	for u, nbrs in ((n, Gr[n]) for n in S):
		c.update((u, v) for v in nbrs if v in T)
	return c


def setCapacities(graph):
	retGraph = nx.DiGraph()
	for (u, v) in graph.edges():
		retGraph.add_edge(u, v, weight=int(round((1 / graph[u][v]['weight']) * 100)))
	return retGraph


def eAlgorithm(G, target, k, source_node, verbose=False):
	Gc = setCapacities(G)
	n = len(Gc.nodes())

	step = 1
	alfa = 0
	source = source_node

	old_S_len = -1

	while True:
		G_alfa = createGAlpha(Gc, alfa, target)

		S, T = computePartitions(G_alfa, source, target)

		if len(S) != old_S_len:
			if verbose:
				print("\nAlpha ", alfa)
				print("|S| =", len(S), "|T| =", len(T))
				print("Alpha ", alfa)

			old_S_len = len(S)

		if len(S) <= k:
			break
		alfa += step

	cuts = computeCutset(Gc, S, T)
	monitor_set = chooseMs(cuts, source, target)
	return monitor_set, len(S)
