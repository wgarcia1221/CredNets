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
graph = GG.WeightedDirectedGraph()
#graph from example but still have to give nodes a value attribute
graph.addNode("A")
#your node can be whatever you want so a tuple would work but an attribute
#in the class would also work
#add attribute to the node rather than passing in the node as a tuple 
graph.addNode(("B", 2))
graph.addNode("C")
graph.addNode("D")
graph.addNode("E")
graph.addEdge("A","D",2)
graph.addEdge("D","E",4)
graph.addEdge("A",("B", 2),2)
graph.addEdge(("B", 2),"C",2)
graph.addEdge("C","A",2)

#I need a function to see the graph or get info on the graphs and the edges
def graphSummary(graph3):
    #how do I access edge weight?
    print(graph.weights)
    print(graph.edges)
    print(graph.nodes)

# def findCycle(graph, node, visited, stack):
#     # Mark current node as visited and  
#         # adds to recursion stack 
#         visited[node] = True
#         stack[node] = True
#         # Recur for all neighbours 
#         # if any neighbour is visited and in  
#         # recStack then graph is cyclic 
#         for neighbour in graph.nodes[node]: 
#             if visited[neighbour] == False: 
#                 if graph.isCyclicUtil(neighbour, visited, stack) == True: 
#                     return True
#             elif stack[neighbour] == True: 
#                 return True
  
#         # The node needs to be poped from  
#         # recursion stack before function ends 
#         stack[node] = False
#         return False

# # Returns true if graph is cyclic else false 
# def isCyclic(graph): 
#     visited = [False] * graph.edges
#     stack = [False] * graph.edges
#     for node in range(graph.nodes): 
#         if visited[node] == False: 
#             if graph.isCyclicUtil(node,visited,stack) == True: 
#                 return True
#     return False

#function that compresses the graph
# def compressGraph(cycle, amount):
#     all_banks = True
#     if all_banks is True:
#         #compress the graph
#         for e in cycle.edges:
            #cycle.weights.get() -= amount
            #return the graph after modifying edge weights
            #return cycle
    
    # if it does not get in if statement, you return the old graph
    #return cycle

# graphSummary(graph)
# if isCyclic(graph) == 1: 
#     print("Graph has a cycle")
# else: 
#     print("Graph has no cycle")
# compressGraph(graph, 1)
graphSummary(graph)