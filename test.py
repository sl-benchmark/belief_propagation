import os
os.chdir('../')
import networkx as nx
import source_loc as sl
import scipy.stats as st
import random
from datetime import datetime
random.seed(datetime.now())


graph=nx.path_graph(20)
dist=st.expon()
source=10
infected = nx.single_source_dijkstra_path_length(graph, source)
obs=[0,19]
obs_time = dict((k,v) for k,v in infected.items() if k in obs)
res=sl.belief_propagation(graph, obs_time, dist)[1]
rank=-1
for i in range(20):
    if res[i][0]==source:
        rank=i
print("Path on 20 nodes, source rank ",rank)


graph=nx.erdos_renyi_graph(20,0.5)
dist=st.expon(0.5)
source=10
infected = nx.single_source_dijkstra_path_length(graph, source)
obs=[0,19,1,2,8]
obs_time = dict((k,v) for k,v in infected.items() if k in obs)

res=sl.belief_propagation(graph, obs_time, dist)[1]
rank=-1
for i in range(20):
    if res[i][0]==source:
        rank=i
print("G(20,0.5), source rank ",rank)
