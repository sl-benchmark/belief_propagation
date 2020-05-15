import os
os.chdir('../')
import networkx as nx
import numpy as np
#import source_loc as sl
import scipy.stats as st
import scipy.sparse as sp
import random
import operator
from datetime import datetime
random.seed(datetime.now())


def psi_message(fact,n, g,T,lam,Omatrix,Omatrix2,obs_time,N):
    Prod1=np.zeros(T)
    Prod2=np.zeros(T)
    if fact[1] in obs_time: # taking the observations into account
        Prod1[int(obs_time[fact[1]])]=1
        Prod2[int(obs_time[fact[1]])]=1
    else:
        Prod1=np.ones(T)
        Prod2=np.ones(T)
    
    for fact2 in g.neighbors(fact):
        if fact2==n: #the node to which we are sending the message to is excluded
            continue
        msg = g.nodes[fact2]['msg'][fact]
        Prod1=np.multiply(Prod1,msg[0,:]+msg[1,:])
        Prod2=np.multiply(Prod2,msg[0,:])

    return np.outer(np.array([1,1]),Prod1)-np.outer(np.array([1,0]),Prod2) 
    
def phi_message(fact,n, g,T,lam,Omatrix,Omatrix2,obs_time,N):
    i = n[1] #n is of the form ("psi",i)
    e = fact[1] #fact is of the form ("phi",e)
    j = e[1] if (e[0]==i) else e[0]
    
    msg = g.nodes[("psi",j)]['msg'][fact]
    return np.reshape(Omatrix.dot(np.reshape(msg,2*T)),(2,T)) # inner product with Omatrix

def ph2_message(fact,n, g,T,lam,Omatrix,Omatrix2,obs_time,N):
    i = n[1] #n is of the form ("psi",i) or ("pri",i)
    e = fact[1] #fact is of the form ("phi",e)
    j = e[1] if (e[0]==i) else e[0]
    
    if j<N:
        msg = g.nodes[("psi",j)]['msg'][fact]
        return np.reshape(Omatrix2.dot(np.reshape(msg,2*T)),(2,T)) # inner product with Omatrix2        
    else:
        msg = g.nodes[("pri",j)]['msg'][fact]
        return np.reshape(Omatrix2.T.dot(np.reshape(msg,2*T)),(2,T)) # inner product with Omatrix2


def genO(typ,T,dist):
    data=[]
    row =[]
    col =[]
    for ti in range(T):
        for pi in range(2):
            for tj in range(T):
                for pj in range(2):
                    tmp = -1
                    if typ==1:
                        tmp=chi(ti, tj, pi, pj,dist)
                    else:
                        tmp=chi2(ti, tj, pi, pj,dist)
                    if tmp:
                        data.append(tmp)
                        col.append(ti+T*pi)
                        row.append(tj+T*pj)
    mat= sp.coo_matrix((data,(row,col)), shape=(2*T,2*T))
    return mat


def chi(ti, tj, pi, pj,dist): #ti: i's belief of its infection time sent to j, pi: i's belief of whether j is a parent or not
    if (pi==0) and (pj==0): # neither is the parent of the other
        if ti==tj: # in this case they can never be each others parents
            return 1
        else:
            return 1-dist.cdf(np.abs(ti-tj)) # we must assure that the 1st one didn't infect the second
    elif (pi-pj)*(ti-tj)>0: #if pi>pj we need ti>tj and vica versa
        return dist.cdf(np.abs(ti-tj))-dist.cdf(np.abs(ti-tj)-1)
    else:
        return 0


def chi2(ti, tj, pi, pj,dist): #only j can infect i
    if (pj==1): #i cannot be a parent of j
        return 0
    if (pi==0): #j is not a parent of i
        return 1
    elif (ti==tj): #if j is a parent is must immidiately infect
        return 1
    else: 
        return 0
        
def pri_message(fact,n,g,T,lam,Omatrix,Omatrix2,obs_time,N):
    tmp = np.vstack([np.ones(T),np.zeros(T)])*lam
    tmp[0,T-1]=1
    return tmp

def get_marginal(n,g,T,lam,Omatrix,Omatrix2):
    Prod1=psi_message(('psi',n),-1,g,T,lam,Omatrix,Omatrix2)[0]
    return Prod1/np.sum(Prod1)

def belief_propagation(graph,obs_time,dist):
    ########################### Init graph
    g = nx.Graph()
    N=graph.number_of_nodes()
    T=int(2*max(dist.mean()*N,max(obs_time.values())-min(obs_time.values())))
    lam = 0.01/N
    nodes = list(graph.nodes())
    source_est_list = []
    source_est_last = -1
    source_est_last_num = 0


    for key, val in obs_time.items():
        obs_time[key]+=int(min(obs_time.values())+dist.mean()*N/2)
    print(obs_time)
    
    for n in graph.nodes():
        g.add_node(("psi",n), typ='factor', fun=psi_message)
        g.add_node(("ph2",(n,n+N)), typ='factor', fun=ph2_message)
        g.add_node(("pri",n+N), typ='factor', fun=pri_message)    
        g.add_path([("psi",n),("ph2",(n,n+N)),("pri",n+N)])
    
        
    g.add_nodes_from([("phi",e) for e in graph.edges()], typ='factor', fun=phi_message)
    g.add_edges_from([(("psi",e[i]),("phi",e)) for e in graph.edges() for i in [0,1] ])
    
    msgs = {n:{'msg':{n2: np.ones((2,T)) for n2 in g.neighbors(n)}} for n in g.nodes()}
    nx.set_node_attributes(g,msgs)
    
    ##################### Running BP
    converged = False
    Omatrix = genO(1,T,dist)
    Omatrix2 = genO(2,T,dist)
    count=0
    
    while (not converged) and (count<100):
        converged = True
        count+=1
        order = [x[1] for x in nx.bfs_edges(g,("psi",nodes[np.random.randint(len(graph.nodes()))]) )] #message scheduling
        for rev in range(2):
            order.reverse()
            for fact in order:
                for n,old_msg in g.nodes[fact]['msg'].items():
                    new_msg = g.nodes[fact]['fun'](fact,n,g,T,lam,Omatrix,Omatrix2,obs_time,N)
                    new_msg =new_msg/np.sum(new_msg)
                    if not np.isclose(new_msg,old_msg).all():
                        converged=False
                    mu = 1 #min(1,10/count)
                    g.nodes[fact]['msg'][n]=mu*new_msg + (1-mu)*g.nodes[fact]['msg'][n]
                
        print("", count, " steps")
        source_est = -1
        maxV=0
        marg=0
        source_est_list.append([])
        for n in graph.nodes():
            marg = np.multiply(g.node[('pri',n+N)]['msg'][('ph2',(n,n+N))],g.node[('ph2',(n,n+N))]['msg'][('pri',n+N)])[0,:]
            marg=marg/np.sum(marg)
            source_est_list[count-1].append(np.sum(marg[0:-1]))
        #    plt.figure()
        #    plt.bar(range(len(marg)),marg)
        #    plt.title(str(n)+" " +str(np.sum(marg[0:-1])))
            if np.sum(marg[0:-1])>maxV:
                maxV=np.sum(marg[0:-1])
                source_est = n

        if (source_est == source_est_last):
            source_est_last_num +=1
            if (source_est_last_num == 10):
                converged = True
        else:
            source_est_last_num = 0
            source_est_last = source_est
        print("Source estimate: ",source_est, " (", maxV,")")
    scores = dict(zip(list(graph.nodes),np.mean(np.array(source_est_list)[max(0,len(source_est_list)-10):,:],0)))
    scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    print(scores)
    return (scores[0][0],scores)

