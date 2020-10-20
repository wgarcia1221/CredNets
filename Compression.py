from Graphs import WeightedDirectedGraph, PathError
import GraphGenerators as GG

from numpy import array, fill_diagonal
import numpy.random as R
from random import choice, random, randint, uniform

'''
Notes: 
Research Questions:
    When is this portfolio compression socially beneficial?
    What incentives do banks have to agree?
-Compression can only occur when all banks agree to it

What does the model need: 
Graph needs to be directed and weighted
Model where the nodes are banks
The nodes have an attribute that stores the external assets of banks
What can be payed depends on assets and liabilities
Is the external asset component of the node greater than or equal to what
has to be paid. If so, the transaction can occur and the bank is not in default
Equity needs to be calculated and it refers to money going in and money that you
have minus what is outgoing
This equity calculation helps determine social impact of portfolio compression
Default Cost Parameters represent the cost of

#cycle is represented by list of nodes and the order that they are in
'''
#generate graph similar to the one in the video
#three nodes in the graph
n = 3
graph = GG.EmptyGraph(n)
#edge directions can be random
graph2 = GG.RandomEdgeDirections(graph)
#edges do need weights
graph3 = GG.AddWeights(graph)
graph3.addEdge(1,2,1)
graph3.addEdge(1,0,4)
graph3.addEdge(0,2,4)
graph3.addEdge(1,2,2)
graph3.addEdge(2,0,1)
print(graph3.weights)
#I need a function to see the graph or get info on the graphs and the edges
def graphSummary(graph3):
    #how do I access edge weight?
    print(graph3.edges)
    print(graph3.nodes)
    for e in graph.edges:
        print("This is edge number " + str(e))
    print("This is the set of nodes " + str(graph.nodes))

#function that compresses the graph
def compressGraph(cycle, amount):
    all_banks = True
    if all_banks is True:
        #compress the graph
        for e in cycle.edges:
            #cycle.weights.get() -= amount
            #return the graph after modifying edge weights
            return cycle
    
    # if it does not get in if statement, you return the old graph
    return cycle

graphSummary(graph3)
# compressGraph(graph, 1)
# graphSummary(graph)