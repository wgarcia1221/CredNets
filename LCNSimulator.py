import CreditNetworks as CN

from argparse import ArgumentParser
import sys
import json

def read_json(json_folder):
	with open("liquiditydefault.json") as f:
		simulator_input = json.load(f)
    config = simulator_input["configuration"]
    parameters = {}
	parameters["json_folder"] = json_folder
	parameters["experiments"] = int(config["experiments"])
	parameters["sims_per_sample"] = int(config["sims_per_sample"])
    parameters["events"] = int(config["events"])
	parameters["nodes"] = int(config["nodes"])
	parameters["c"] = int(config["c"])
	parameters["dlow"] = int(config["dlow"])
	parameters["dhigh"] = float(config["dhigh"])
	parameters["plow"] = float(config["plow"])
	parameters["phigh"] = float(config["phigh"])
	parameters["social_network"] = str(config["social_network"])
	parameters["price"] = getattr(sys.modules[__name__], config["price"])
	parameters["prevent_zeros"] = True if config["prevent_zeros"] == "True" \
									else False
	return parameters

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("jsons", type=str)
	parser.add_argument("samples", type=int)
	args = parser.parse_args()
	parameters = read_json(args.jsons)
	parameters["samples"] = args.samples
	return parameters


def run_simulator(parameters):
	n = len(parameters["strategies"])
	payoffs = dict(zip(range(n), [0]*n))
	for sim in range(parameters["sims_per_sample"]):
		matrices = CN.InitMatrices(parameters)
		crednet = CN.InitCrednet(matrices, parameters)
		sim_payoffs = CN.SimulateCreditNetwork(crednet, parameters, **matrices)
		for agent, value in sim_payoffs.items():
			payoffs[agent] += value
	for agent in range(n):
		payoffs[agent] /= parameters["sims_per_sample"]
	return payoffs


def main():
	parameters = parse_args()
	for i in range(parameters["samples"]):
		payoffs = run_simulator(parameters)
		write_payoffs(payoffs, parameters, str(i))


if __name__ == "__main__":
	main()
