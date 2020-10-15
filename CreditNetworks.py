from Graphs import WeightedDirectedGraph, PathError
import GraphGenerators as GG
from Strategies import AgentStrategies, BankPolicies

from numpy import array, fill_diagonal
import numpy.random as R
from random import choice, random, randint, uniform

from argparse import ArgumentParser
import sys
import json

parser = ArgumentParser()
parser.add_argument("json_file", type=str)
args = parser.parse_args()

with open(args.json_file) as f:
	parameters = json.load(f)
print(parameters)

class CreditError(Exception):
	def __init__(self):
		Exception.__init__(self, 'insufficient credit')


class CreditNetwork(WeightedDirectedGraph):
	def __init__(self, nodes=[], weightedEdges=[]):
		WeightedDirectedGraph.__init__(self, nodes, weightedEdges)

	def capacity(self, path):
		"""
		Determine the minimum weight along a path.

		The path should be represented as a list of nodes. If given an edge,
		capacity() will return the edge's weight.

		Returns inf for degenerate paths with length < 2.
		"""
		minCapacity = float("inf")
		for src, dst in zip(path, path[1:]):
			minCapacity = min(minCapacity, self.weights[(src, dst)])
		return minCapacity

	def makePayment(self, sender, receiver, amount):
		"""
		Transfer an IOU for <amount> from <sender> to connected <receiver>.

		There must be an edge with weight >= amount from the receiver
		to the sender; otherwise the assert fails.

		The credit edge from receiver to sender is debited by amount, while the
		credit edge from sender to receiver is increased by amount.
		"""
		assert amount > 0
		if not self.adjacent(receiver, sender) or \
				self.weights[receiver, sender] < amount:
			raise CreditError()
		if not self.adjacent(sender, receiver):
			self.addEdge(sender, receiver, amount)
		else:
			self.weights[(sender, receiver)] += amount
		self.weights[(receiver, sender)] -= amount
		if self.weights[(receiver, sender)] == 0:
			self.removeEdge(receiver, sender)

	def routePayment(self, sender, receiver, amount):
		"""
		Transfer IOUs through the credit network from sender to receiver.

		There must be directed paths from reciever to sender with total capacity
		of at least amount. If not, a CreditError is raised.
		"""
		remaining = amount
		while remaining > 0:
			try:
				path = self.shortestPath(receiver, sender)
			except PathError:
				self.routePayment(receiver, sender, amount - remaining)
				raise CreditError()
			capacity = self.capacity(path)
			for src, dst in zip(path[1:], path):
				self.makePayment(src, dst, min(capacity, remaining))
			remaining = max(remaining - capacity, 0)


def SimulateCreditNetwork(CN, params, DP, TR, BV, SC):
	"""
	CN - credit network
	DP - default probability array
	TR - transaction rate matrix
	BV - buy value matrix
	SC - sell cost matrix
	price - function to determine a price from value and cost
	events - number of transactions to simulate
	"""
	price = params["price"]
	events = params["events"]
	strategies = params["strategies"]
	prevent_zeros = params["prevent_zeros"]

	payoffs = dict([(n,0.) for n in CN.nodes])
	defaulters = filter(lambda n: R.binomial(1, DP[n]), CN.nodes)

	# If all agents with the same strategy default, we'll get bad payoff data
	while prevent_zeros:
		prevent_zeros = False
		for strat in set(strategies):
			agents = filter(lambda a: strategies[a]==strat, CN.nodes)
			if all([a in defaulters for a in agents]):
				prevent_zeros = True
				defaulters = filter(lambda n: R.binomial(1, DP[n]), CN.nodes)
				break

	for d in defaulters:
		for n in CN.nodes:
			if CN.adjacent(n, d):
				payoffs[n] -= CN.weights[(n, d)]
		CN.removeNode(d)
		del payoffs[d]

	#events is used here but for what, it is used to determine agent strategy so that they don;t have the same strategy
	m = R.multinomial(events, array(TR.flat))
	l = TR.shape[0]
	transactors = sum([[(i/l,i%l)]*m[i] for i in range(l**2)], [])
	R.shuffle(transactors)
	for b,s in transactors:
		try:
			assert b in CN.nodes and s in CN.nodes
			CN.routePayment(b, s, price(BV[b,s], SC[b,s]))
		except (AssertionError, CreditError):
			continue
		payoffs[b] += BV[b,s]
		payoffs[s] -= SC[b,s]
	return payoffs


def InitMatrices(parameters):
	"""
	The following parameters are required:
	nodes...a list with length = number of nodes in the credit network
	def_alpha...alpha parameter for default probability beta-distribution
	def_beta....beta parameter for default probability beta-distribution
	rate_alpha..alpha parameter for transaction rate pareto-distribution
	min_value...minimum for buy value uniform-distribution
	max_value...maximum for buy value uniform-distribution
	min_cost....minimum for sell cost uniform-distribution
	max_cost....maximum for sell cost uniform-distribution
	"""
	n = len(params["nodes"])
	matrices = dict()
	matrices["DP"] = R.beta(params["def_alpha"], params["def_beta"], n)
	matrices["TR"] = R.pareto(params["rate_alpha"], [n]*2)
	fill_diagonal(matrices["TR"], 0)
	matrices["TR"] /= matrices["TR"].sum()
	matrices["BV"] = R.uniform(params["min_value"], params["max_value"], [n]*2)
	fill_diagonal(matrices["BV"], 0)
	matrices["SC"] = R.uniform(params["min_cost"], params["max_cost"], [n]*2)
	fill_diagonal(matrices["SC"], 0)
	return matrices


def InitCrednet(matrices, parameters):
	"""
	The following parameters are required:
	strategies......list of strategies by which agents issue credit
	social_network..1-argument function to create a social network
	num_banks.......number of banks to simulate (usually 0 or 1)
	bank_policy.....the policy used to create credit edges involving banks

	plus required parameters of AgentStrategies and BankPolicies
	"""
	n = len(params["strategies"])
	social_network = getattr(GG, params["social_network"])(n)
	#can't do this so determine what these do and recreate them
	AS = AgentStrategies(matrices, social_network, params)
	BP = BankPolicies(matrices, social_network, params)
	nodes = range(-params["num_banks"], n)
	edges = sum([AS.get_strategy(s)(agent) for agent,s in enumerate( \
			params["strategies"])] + [BP.get_policy(params["bank_policy"])( \
			bank) for bank in range(-params["num_banks"],0)], [])
	print(edges)
	return CreditNetwork(nodes, edges)

def SimulateLCN(CN):
	#implement warm up 
	#I dont remember how to do this
	window_size = parameters["steps"]	
	epsilon = 0.002
	success = []
	#should I load capacities here?

	
	for i in range(window_size):
		success_count = 0 
			#Do I need to randomly two nodes?
			#how many times does this transaction need to be run? 
			#check if path exists from s to t 
			#I'm not sure if I need these but it seems that this call checks if there is a path
		total_transactions += 1
		try: 
			# if yes route payment from s to t and modify edges along that path
			CN.routePayment(CN, sender, receiver, amount)
			success_count += 1
		except:
			#else count transaction as a failure
			pass
		success[i] = success_count/total_transactions



def InitLiqCredNet(parameters):
	#social network is passed into the json to determine which graph is being generated depending on the 
	#simulation being run
	#differentiate per experiment 
	# for experiment 1 where network density is being tested 
	if (args.json_file == "E1.json"):
		num_nodes = parameters["num_nodes"]
		if (parameters["social_network"] == "ErdosRenyiGraph"):
			p = random.uniform(parameters["edge_probability_low"], parameters["edge_probability_high"])
			graph = GG.ErdosRenyiGraph(num_nodes, p)

		if (parameters["social_network"] == "BarabasiAlbertGraph"):
			d = random.randint(parameters["edges_per_node_low"], parameters["edges_per_node_high"])
			graph = GG.BarabasiAlbertGraph(num_nodes, d)
		
		return CreditNetwork(graph.nodes, graph.edges)
	
	# for experiment 2 where credit capacity is being tested:
	elif (args.json_file == "E2.json"):
		num_nodes = parameters["num_nodes"]
		
		if (parameters["social_network"] == "ErdosRenyiGraph"):
			p = parameters["edge_probability"]
			graph = GG.ErdosRenyiGraph(num_nodes, p)

		if (parameters["social_network"] == "BarabasiAlbertGraph"):
			d = parameters["d"]
			graph = GG.BarabasiAlbertGraph(num_nodes, d)
		
		return CreditNetwork(graph.nodes, graph.edges)

	# for experiment 3 where varying network size is being tested:
	elif (args.json_file == "E3A.json"):
		
		if (parameters["social_network"] == "ErdosRenyiGraph"):
			num_nodes = random.randint(parameters["nodes_low"], parameters["nodes_high"])
			p = params["p"]
			graph = GG.ErdosRenyiGraph(num_nodes, p)

		if (parameters["social_network"] == "BarabasiAlbertGraph"):
			d = parameters["d"]
			graph = GG.BarabasiAlbertGraph(num_nodes, d)

	return CreditNetwork(graph.nodes, graph.edges)

	elif (args.json_file == "E3B.json"):
		if (parameters["social_network"] == "ErdosRenyiGraph"):
			num_nodes = random.randint(parameters["nodes_low"], parameters["nodes_high"])
			#There are two different experiments that determine what p is going 
			p = 1 / (num_nodes / 10)
			graph = GG.ErdosRenyiGraph(num_nodes, p)
		
		if (parameters["social_network"] == "BarabasiAlbertGraph"):
			d = parameters["d"]
			graph = GG.BarabasiAlbertGraph(num_nodes, d)

	return CreditNetwork(graph.nodes, graph.edges)

# def test(parameters):
# 	print(5)
# 	n = len(parameters["strategies"])
# 	print(n)

def main():
	
# 	data = []
# 	parameters = parse_args()
# 	for i in range(1, parameters["runs"]):
# 		#run and compute average and standard deviation of steady state success probability
# 		success = simulateLCN(CN)
# 		data += success
# 	avg = sum(data)/len(data)
# 	stdev = np.std(data)
# 	print("The average in success rate over 100 runs is" + str(avg))
# 	print("The standard deviation in success rate over 100 runs is" + str(stdev))

# test(parameters)
	main()