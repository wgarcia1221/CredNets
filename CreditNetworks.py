from Graphs import WeightedDirectedGraph, PathError
import GraphGenerators as GG
from Strategies import AgentStrategies, BankPolicies

from numpy import array, fill_diagonal
import numpy.random as R
from random import choice, random, randint, uniform


#Notes Section: 
#1. You want to map the number of nodes to a specific number
#2. Rewrite init.CredNet in this file
#3. Rewrite CN.simulateCreditNetwork also in this file
#4. Rewrite init.Matrices

# I definitely need this social network variable
# "social_network":"ErdosRenyiGraph",

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
	window_size = 1000
	epsilon = 0.002
	success = []
	while (abs(success[i] - success[i-1]) < epsilon):
		for i in range(window_size):
			success_count = 0 
			total_transactions = 0 
			success_rate = success_count / total_transactions
			#choose node pair s,t
			for :
				for n in CN.nodes:
			#check if path exists from s to t 
			#I'm not sure if I need these but it seems that this call checks if there is a path
					total_transactions += 1
					try: 
						# if yes route payment from s to t and modify edges along that path
						#make payment modifies edge capacities
						CN.routePayment(self, n, )
						success_count += 1
					except:
						#else count transaction as a failure
						pass



def InitLiqCredNet(parameters):
	#social network is passed into the json to determine which graph is being generated depending on the 
	#simulation being run
	#differentiate per experiment 
	
	# for experiment 1 where network density is being tested 
	if (parameters["experiment"] == 1):
		num_nodes = 200
		plow = 0.18
		phigh = 0.45
		dlow = 18
		dhigh = 45
		nodes = range(-1, num_nodes)
		#the difference between these is how the p and the d are passed in to get the range for the edges for netwroek density
		#why is this error here?
		if (parameters["social_network"] == "ErdosRenyiGraph"):
			p = random.uniform(plow, phigh)
			num_edges = (nodes - 1) * p
			edges = []
			for i in range(num_edges):
				edges += parameters["c"]
			
			#should create graph in question but I am not sure how to give it attributes that it needs
			graph = GG.ErdosRenyiGraph(num_nodes, p)

		if (parameters["social_network"] == "BarabasiAlbertGraph"):
			d = random.randint(dlow, dhigh)
			num_edges = 2 * d 
			edges = []
			for i in range(num_edges):
				edges += parameters["c"]
			graph = GG.BarabasiAlbertGraph(num_nodes, d)
		
		return CreditNetwork(nodes, edges)
	
	# for experiment 2 where credit capacity is being tested:
	elif (parameters["experiment"] == 2):
		nodes = range(-1, num_nodes)

		if (parameters["social_network"] == "ErdosRenyiGraph"):
			p = parameters["p"]
			num_edges = (nodes - 1) * p
			edges = []
			for i in range(num_edges):
				edges += parameters["c"]

		if (parameters["social_network"] == "BarabasiAlbertGraph"):
			d = parameters["d"]
			num_edges = 2 * d 
			edges = []
			for i in range(num_edges):
				edges += parameters["c"]

		return CreditNetwork(nodes, edges)

	# for experiment 3 where varying network size is being tested:
	elif (parameters["experiment"] == 3):
		nodesl = 20
		nodesh = 500
		
		if (parameters["social_network"] == "ErdosRenyiGraph"):
			nodes = random.randint(nodesl, nodesh)
			#There are two different experiments that determine what p is going 
			p = params["p"]
			num_edges = (nodes - 1) * p
			edges = []
			for i in range(num_edges):
				edges += parameters["c"]
		
		if (params["social_network"] == "BarabasiAlbertGraph"):
			d = params["d"]
			num_edges = 2 * d 
			edges = []
			for i in range(num_edges):
				edges += parameters["c"]

	return CreditNetwork(nodes, edges)

	elif (parameters["experiment"] == 4):
		nodesl = 20
		nodesh = 500

		if (parameters["social_network"] == "ErdosRenyiGraph"):
			nodes = random.randint(nodesl, nodesh)
			
			#There are two different experiments that determine what p is going 
			p = random.uniform(params["plow"], params["phigh"])

			num_edges = (nodes - 1) * p
			edges = []
			for i in range(num_edges):
				edges += parameters["c"]
		
		if (params["social_network"] == "BarabasiAlbertGraph"):
			d = params["d"]
			num_edges = 2 * d
			edges = []
			for i in range(num_edges):
				edges += parameters["c"]

	return CreditNetwork(nodes, edges)

def Test(parameters):
	print(5)
	read_json(jsons)
	n = len(parameters["strategies"])
	print(n)


Test(parameters)