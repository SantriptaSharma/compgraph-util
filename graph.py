import string, random
from typing import List

NODE_TYPES = {
    "MATMUL": 0,
    "COMBINE": 1,
    "SPLIT": 2,
    "DATA": 3,
    "SCALE": 4,
    "PASSTHROUGH": 5,
}

CHARSET = string.ascii_lowercase + string.digits

def generate_node_id():
	return ''.join(random.choice(CHARSET) for _ in range(16))

class GraphNode:
	def __repr__(self) -> str:
		typename = list(NODE_TYPES.keys())[self.type]
		primary_group = "" if len(self.group_names) == 0 else " " + self.group_names[0]
		data = str(self.data)
		if len(data) > 40:
			data = f"{data[0:40]}..."

		return f"{self.id}: {typename}{primary_group} ({data}) -> {self.output_shape}"

	def __str__(self) -> str:
		return repr(self)

	def __init__(self, group_names = [], n_type = 0, data = {}, prev = [], output_shape = (1, 1), bandwidth_out = 16, capacity = 16, compute_units = 1) -> None:
		self.type = n_type
		self.prev = prev
		self.group_names = group_names
		self.id = generate_node_id()
		self.bandwidth = bandwidth_out
		self.capacity = capacity
		self.compute_units = compute_units

		self.data = data
		self.output_shape = output_shape

	def validate(self) -> (bool, List[str]):
		""" Checks whether the given node has valid data parameters and output shape, and calculate dynamic parameters """
		errors = []
		
		if len(self.prev) == 0 and self.type not in [NODE_TYPES["DATA"]]:
			errors.append(f"Node has no inputs: {self}")
		
		valid = True

		if self.type == NODE_TYPES["DATA"]:
			if len(self.prev) != 0:
				errors.append(f"DATA nodes cannot have any inputs: {self}")
		elif self.type == NODE_TYPES["COMBINE"]:
			if "axis" not in self.data or self.data["axis"] not in ["row", "col"]:
				errors.append(f"COMBINE nodes must have an axis parameter (row or col): {self}")
			else:
				if self.data["axis"] == "row":
					shape = self.prev[0].output_shape
					for node in self.prev:
						if shape[1] != node.output_shape[1]:
							errors.append(f"inputs to row-wise COMBINE nodes must have the same number of columns: {self}")
							break
					self.output_shape = (sum([node.output_shape[0] for node in self.prev]), shape[1])
				else:
					shape = self.prev[0].output_shape
					for node in self.prev:
						if shape[0] != node.output_shape[0]:
							errors.append(f"inputs to column-wise COMBINE nodes must have the same number of rows: {self}")
							break
					self.output_shape = (shape[0], sum([node.output_shape[1] for node in self.prev]))
				
		elif self.type == NODE_TYPES["SPLIT"]:			
			if "axis" not in self.data or self.data["axis"] not in ["row", "column"]:
				valid = False
				errors.append(f"SPLIT nodes must have an 'axis' parameter (row or col): {self}")
			
			if "ways" not in self.data or not isinstance(self.data["ways"], int) or self.data["ways"] <= 0:
				valid = False
				errors.append(f"SPLIT nodes must have a positive integer 'ways' parameter: {self}")

			if len(self.prev) != 1:
				valid = False
				errors.append(f"SPLIT nodes must have exactly one input: {self}")
			
			input_shape = self.prev[0].output_shape

			if self.data["axis"] == "row" and input_shape[0] % self.data["ways"] != 0:
				valid = False
				errors.append(f"SPLIT nodes must have an input with a number of rows divisible by the 'ways' parameter: {self}, {input_shape}")
			
			if self.data["axis"] == "col" and input_shape[1] % self.data["ways"] != 0:
				valid = False
				errors.append(f"SPLIT nodes must have an input with a number of columns divisible by the 'ways' parameter: {self}, {input_shape}")

			if valid:
				if self.data["axis"] == "row":
					self.output_shape = (input_shape[0] // self.data["ways"], input_shape[1])
				else:
					self.output_shape = (input_shape[0], input_shape[1] // self.data["ways"])

		elif self.type == NODE_TYPES["MATMUL"]:
			if len(self.prev) != 2:
				valid = False
				errors.append(f"MATMUL nodes must have exactly two inputs: {self}")
			
			transpose_a = "transpose_a" in self.data and self.data["transpose_a"]
			transpose_b = "transpose_b" in self.data and self.data["transpose_b"]

			shape_a = (self.prev[0].output_shape[1], self.prev[0].output_shape[0]) if transpose_a else self.prev[0].output_shape
			shape_b = (self.prev[1].output_shape[1], self.prev[1].output_shape[0]) if transpose_b else self.prev[1].output_shape

			if shape_a[1] != shape_b[0]:
				valid = False
				errors.append(f"MATMUL nodes must have inputs with compatible shapes: {self}, {self.prev[0].output_shape} x {self.prev[1].output_shape}")
			
			if valid:
				self.output_shape = (shape_a[0], shape_b[1])
		elif self.type == NODE_TYPES["SCALE"]:
			if len(self.prev) != 1:
				valid = False
				errors.append(f"SCALE nodes must have exactly one input: {self}")
			
			if valid:
				self.output_shape = self.prev[0].output_shape
		elif self.type == NODE_TYPES["PASSTHROUGH"]:
			if len(self.prev) != 1:
				valid = False
				errors.append(f"PASSTHROUGH nodes must have exactly one input: {self}")
			
			if valid:
				self.output_shape = self.prev[0].output_shape
		else:
			return (False, "Invalid node type")
		
		return (len(errors) == 0, errors)

	def cost(self) -> int:
		"""	Returns the cost of this computation step in terms of the number of MAC operations required at this step """
		if self.type == NODE_TYPES["MATMUL"]:
			transpose_a = "transpose_a" in self.data and self.data["transpose_a"]
			transpose_b = "transpose_b" in self.data and self.data["transpose_b"]

			shape_a = (self.prev[0].output_shape[1], self.prev[0].output_shape[0]) if transpose_a else self.prev[0].output_shape
			
			return self.output_shape[0] * self.output_shape[1] * shape_a[1]
		elif self.type == NODE_TYPES["SCALE"]:
			return self.output_shape[0] * self.output_shape[1]
		
		return 0
	
	def apply_namespace(self, namespace: str):
		"""	Applies a namespace to this node and all of its children """
		for i in range(len(self.group_names)):
			if not self.group_names[i].startswith(namespace):
				self.group_names[i] = f"{namespace}_{self.group_names[i]}"
		
		for node in self.prev:
			node.apply_namespace(namespace)


class CompGraph:
	def __init__(self, end: GraphNode):
		self.starts = []
		self.end = end
		self.nodes = {}
		self.next = {end.id: []}
		self.node_costs = {}
		self.node_groups = {}
		self.total_cost = -1

		visited_set = set()
		queue = [end]

		while len(queue) != 0:
			node = queue.pop(0)

			if node in visited_set:
				continue

			visited_set.add(node)

			if len(node.prev) == 0:
				self.starts.append(node)

			if len(node.group_names) > 0:
				for group_name in node.group_names:
					if group_name not in self.node_groups:
						self.node_groups[group_name] = []
					self.node_groups[group_name].append(node)
			
			while node.id in self.nodes:
				node.id = generate_node_id()
			
			self.nodes[node.id] = node

			for prev in node.prev:
				if prev.id not in self.next:
					self.next[prev.id] = []
				
				self.next[prev.id].append(node)
				
				queue.append(prev)

		validation_queue = self.toposort()

		error_list = []
		for node in validation_queue:
			(_, errors) = node.validate()
			error_list.extend(errors)

		if len(error_list) > 0:
			final_err = "\n".join(error_list)
			raise Exception(f"Invalid graph\n{final_err}")

	def toposort(self):
		visited_set = set()
		topo_sorted = []
		
		def topo_rec(id):
			visited_set.add(id)

			for next in self.next[id]:
				if next.id not in visited_set:
					topo_rec(next.id)

			topo_sorted.append(self.nodes[id])

		for id in self.nodes:
			if id not in visited_set:
				topo_rec(id)
		
		return topo_sorted[::-1]

	def compute_costs(self) -> int:
		for node in self.nodes.values():
			self.node_costs[node.id] = node.cost()
		
		total_cost = sum(self.node_costs.values())
		self.total_cost = total_cost
		return total_cost
	
	def groups(self) -> List[str]:
		return list(self.node_groups.keys())
	
	def compute_group_cost(self, group_name: str) -> int:
		cost = 0
		for node in self.node_groups[group_name]:
			n_cost = node.cost()
			self.node_costs[node.id] = n_cost
			cost += n_cost
	      
		return cost

	def compute_dependency(self):
		pass

	def compute_latency(self) -> int:
		""" returns the latency of the graph in terms of the number of cycles required to execute the graph """
		pass

	def export_graph(self, filename):
		""" export graph to a yaml file """
		pass
	
def import_graph(self, filename) -> CompGraph:
	""" import graph from a yaml file"""
	pass

def bert_encoder(first = True) -> GraphNode:
	""" returns a GraphNode hierarchy representing one encoder in the BERT model """
	dim = 1024
	hidden_dim = 4096
	len = 512
	heads = 16
	dh = dim//heads
	
	end = GraphNode(n_type = NODE_TYPES["SCALE"], group_names = ["Output"], prev = [GraphNode(n_type = NODE_TYPES["COMBINE"], data = {"axis": "row"}, group_names = ["Output"])])
	pre_FFN = GraphNode(n_type = NODE_TYPES["SPLIT"], data = {"axis": "row", "ways": len}, prev = [])
	
	FFNs = [
		GraphNode(n_type = NODE_TYPES["MATMUL"], group_names = [f"FFN{i + 1} Output Layer", f"FFN{i + 1}", "FFN"], prev = [
			GraphNode(n_type = NODE_TYPES["MATMUL"], group_names = [f"FFN{i + 1} Hidden Layer", f"FFN{i + 1}", "FFN"], prev = [
				pre_FFN,
				GraphNode(n_type = NODE_TYPES["DATA"], group_names = [f"FFN{i + 1} Hidden Weights", f"FFN{i + 1}", "FFN"], output_shape = (dim, hidden_dim))
			]),
			GraphNode(n_type = NODE_TYPES["DATA"], group_names = [f"FFN{i + 1} Output Weights", f"FFN{i + 1}", "FFN"], output_shape = (hidden_dim, dim))
		])
	for i in range(len)]

	end.prev[0].prev = FFNs

	pre_heads = GraphNode(n_type = NODE_TYPES["DATA" if first else "PASSTHROUGH"], output_shape = (len, dim), group_names = ["Input Sequence", "Input"])
	
	attention_heads = [
		GraphNode(n_type = NODE_TYPES["MATMUL"], group_names = [f"Attention Head {i + 1} Output Layer", f"Attention Head {i + 1}"], prev = [
			GraphNode(n_type = NODE_TYPES["SCALE"], group_names= [f"Attention Head {i + 1} Softmax(*)+Scale", f"Attention Head {i + 1}"], prev = [
				GraphNode(n_type = NODE_TYPES["MATMUL"], group_names = [f"Attention Head {i + 1} KQ", f"Attention Head {i + 1}"], prev = [
					GraphNode(n_type = NODE_TYPES["MATMUL"], group_names = [f"Attention Head {i + 1} Q", f"Attention Head {i + 1}"], prev = [
						pre_heads,
						GraphNode(n_type = NODE_TYPES["DATA"], group_names = [f"Attention Head {i + 1} Q Weights", f"Attention Head {i + 1}"], output_shape = (dim, dh))
					]),
					GraphNode(n_type = NODE_TYPES["MATMUL"], group_names = [f"Attention Head {i + 1} K", f"Attention Head {i + 1}"], prev = [
						pre_heads,
						GraphNode(n_type = NODE_TYPES["DATA"], group_names = [f"Attention Head {i + 1} K Weights", f"Attention Head {i + 1}"], output_shape = (dim, dh))
					])
				], data = {"transpose_b": True})
			]),
			GraphNode(n_type = NODE_TYPES["MATMUL"], group_names = [f"Attention Head {i + 1} V", f"Attention Head {i + 1}"], prev = [
				pre_heads,
				GraphNode(n_type = NODE_TYPES["DATA"], group_names = [f"Attention Head {i + 1} V Weights", f"Attention Head {i + 1}"], output_shape = (dim, dh))
			]),
		])
	for i in range(heads)]

	pre_FFN.prev = [
		GraphNode(n_type = NODE_TYPES["MATMUL"], group_names = ["Self-Attention"], prev = [
			GraphNode(n_type = NODE_TYPES["COMBINE"], group_names = ["Self-Attention"], data = {"axis": "col"}, prev = attention_heads),
			GraphNode(n_type = NODE_TYPES["DATA"], group_names = ["Self-Attention Output Weights", "Self-Attention"], output_shape = (dim, dim))
		])
	]

	start = pre_heads

	return (start, end)


if __name__ == "__main__":
	encoders = 24
	end = None
	cur = None
	for i in range(encoders, 0, -1):
		(encoder_start, encoder_end) = bert_encoder(i == 1)
		encoder_end.apply_namespace(f"Encoder {i}")
		if cur is not None:
			cur.prev = [encoder_end]
		else:
			end = encoder_end

		cur = encoder_start

	graph = CompGraph(end)
	graph.compute_costs()