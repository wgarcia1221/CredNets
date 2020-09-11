from Graphs import WeightedDirectedGraph, PathError
import GraphGenerators as GG
from Strategies import AgentStrategies, BankPolicies

from numpy import array, fill_diagonal
import numpy.random as R
from random import choice

#Notes Section: 
#1. Set up JSON configuration file from scratch
#2. You want to map the number of nodes to a specific number
#3. Rewrite init.CredNet in this file
#4. Rewrite CN.simulateCreditNetwork also in this file
# What do I need from the old configuration file? 
#simulations per sample but what is a sample in this case?
#This variable just seems to be the number of simulations
# "sims_per_sample":100,
# "events":"1000",
# you can't run the init.Matrices without these three so you need them 
# or will my credit net not pass these in. This is a good question to ask. 
# The same goes for the min and max parameters passed 
# "def_alpha":"1",
# "def_beta":"19",
# "rate_alpha":"2",
# "min_value":"1",
# "max_value":"2",
# "min_cost":"1",
# "max_cost":"1",

# "price":"cost",

# I definitely need this social network variable
# "social_network":"ErdosRenyiGraph",

# "num_banks":"0",
# "prevent_zeros":"True"
# WHAT DONT I NEED FROM THE JSON?
# "bank_policy":"agents2_banks10",

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


def InitMatrices(params):
	"""
	The following parameters are required:
	strategies..a list with length = number of nodes in the credit network
	def_alpha...alpha parameter for default probability beta-distribution
	def_beta....beta parameter for default probability beta-distribution
	rate_alpha..alpha parameter for transaction rate pareto-distribution
	min_value...minimum for buy value uniform-distribution
	max_value...maximum for buy value uniform-distribution
	min_cost....minimum for sell cost uniform-distribution
	max_cost....maximum for sell cost uniform-distribution
	"""
	n = len(params["strategies"])
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


#

def InitCrednet(matrices, params):
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
	AS = AgentStrategies(matrices, social_network, params)
	BP = BankPolicies(matrices, social_network, params)
	nodes = range(-params["num_banks"], n)
	edges = sum([AS.get_strategy(s)(agent) for agent,s in enumerate( \
			params["strategies"])] + [BP.get_policy(params["bank_policy"])( \
			bank) for bank in range(-params["num_banks"],0)], [])
	return CreditNetwork(nodes, edges)


def InitLiqCredNet(matrices, params):
	#social network is passed into the json to determine which graph is being generated depending on the 
	#simulation being run
	#differentiate per experiment 
	# 1.
	nodes = params["nodes"]
	n = len(params["strategies"])
	social_network = getattr(GG, params["social_network"])(n)
	AS = AgentStrategies(matrices, social_network, params)
	
	# I'm confused here 
	edges = sum([AS.get_strategy(s)(agent) for agent,s in enumerate( \
			params["strategies"])] + [BP.get_policy(params["bank_policy"])( \
			bank) for bank in range(-params["num_banks"],0)], [])

	#the difference between these is how the p and the d are passed in to get the range for the edges for netwroek density
	#why is this error here?
	if (params["social_network"] == "ErdosRenyiGraph"):
		e = range(params["plow"], params["phigh"])
	

	if (params["social_network"] == "BarabasiAlbertGraph"):
		d = range(params["dlow"], params["dhigh"])
	
	return CreditNetwork(nodes, edges)	
	
	# 2.

	# 3.