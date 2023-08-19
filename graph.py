import string, random
from typing import List
from functools import cache
from math import prod

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

class GraphNamespace:
	def __init__(self, name: str, parent = None):
		self.name = name
		self.parent = parent
		self.children = []

		if self.parent is not None:
			self.parent.children.append(self)

	@cache
	def __repr__(self):
		s = []
		cur = self

		while cur is not None:
			s.append(cur.name)
			cur = cur.parent
		
		return ".".join(s[::-1])
	
	def __str__(self):
		return self.name
	
	@cache
	def traverse_down(self):
		""" returns a list of all children namespaces including self """
		s = [self]
		for child in self.children:
			s.extend(child.traverse_down())
		return s

	@cache
	def traverse(self):
		""" returns a list of all connected namespaces """
		root = self
		while root.parent is not None:
			root = root.parent
		
		return root.traverse_down()

class GraphNode:
	def __repr__(self) -> str:
		typename = list(NODE_TYPES.keys())[self.type]
		name = "" if self.name is None else f" {self.namespace.name + '.' if self.namespace is not None else ''}" + self.name
		data = str(self.data)
		if len(data) > 40:
			data = f"{data[0:40]}..."

		return f"{self.id}: {typename}{name} ({data}) -> {self.output_shape}"

	def __str__(self) -> str:
		return repr(self)

	def __init__(self, name = None, n_type = 0, data = {}, prev = [], output_shape = (1, 1), bandwidth_out = 16, capacity = 16, compute_units = 1, namespace = None) -> None:
		self.type = n_type
		self.prev = prev
		self.name = name
		self.id = generate_node_id()
		self.bandwidth = bandwidth_out
		self.capacity = capacity
		self.compute_units = compute_units
		self.namespace = namespace

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


class CompGraph:
	#TODO: add support for extracting namespaces as subgraphs

	def __init__(self, end: GraphNode):
		# TODO: move work into a compile-ish method for recomputing on changes?
		self.starts = []
		self.topo = []
		self.end = end
		self.next = {end.id: []}
		self.nodes = {}
		self.nodes_by_name = {}
		self.node_costs = {}
		self.node_groups = {}
		self.namespace_nodes = {}
		self.namespaces = {}
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

			if node.name is not None:
				if node.name not in self.nodes_by_name:
					self.nodes_by_name[node.name] = []
				self.nodes_by_name[node.name].append(node)

			if node.namespace is not None:
				qualified = repr(node.namespace)
				registered = qualified in self.namespaces
				self.namespaces[qualified] = node.namespace
				
				if not registered:
					connected = node.namespace.traverse()

					for ns in connected:
						qual_conn = repr(ns)
						self.namespaces[qual_conn] = ns
				
				if qualified not in self.namespace_nodes:
					self.namespace_nodes[qualified] = []
				self.namespace_nodes[qualified].append(node)
			
			while node.id in self.nodes:
				node.id = generate_node_id()
			
			self.nodes[node.id] = node

			for prev in node.prev:
				if prev.id not in self.next:
					self.next[prev.id] = []
				
				self.next[prev.id].append(node)
				
				queue.append(prev)

		validation_queue = self.toposort()
		self.topo = validation_queue

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

	def direct_namespace_cost(self, namespace: GraphNamespace) -> int:
		""" returns the cost of a namespace """
		if repr(namespace) not in self.namespace_nodes:
			return 0
		
		return sum([self.node_costs[node.id] for node in self.namespace_nodes[repr(namespace)]])
	
	def namespace_cost(self, namespace: GraphNamespace) -> int:
		""" returns the cost of a namespace and all its children """
		if repr(namespace) not in self.namespaces:
			return 0
		
		return self.direct_namespace_cost(namespace) + sum([self.namespace_cost(ns) for ns in namespace.children])
	
	def get_namespaces(self) -> List[str]:
		return [repr(ns) for ns in self.namespaces.keys()]
	
	def get_all_nodes_in(self, namespace):
		""" returns all nodes in a namespace and its children """
		if repr(namespace) not in self.namespaces:
			return []
		
		tree = namespace.traverse_down()
		nodes = set()
		for node in tree:
			r = repr(node)
			if r in self.namespace_nodes:
				nodes.update(self.namespace_nodes[r])

		return list(nodes)
	
	def namespace_data(self, namespace: GraphNamespace) -> dict:
		""" returns information about the data flow quantities in the namespace in words """
		if repr(namespace) not in self.namespaces:
			return {"data-in": 0, "data-resident": 0, "data-out": 0, "in_nodes": set(), "out_nodes": set(), "resident": set()}
		
		tree = namespace.traverse_down()
		nodes = self.get_all_nodes_in(namespace)
		
		data_in = data_out = data_resident = 0

		in_nodes = set()
		out_nodes = set()
		resident = set()

		for node in nodes:
			if node.type == NODE_TYPES["DATA"]:
				data_resident += node.output_shape[0] * node.output_shape[1]
				resident.add(node)
			else:
				out = any([n.namespace not in tree for n in self.next[node.id]])

				if out:
					out_nodes.add(node)

				inp = filter(lambda n: n.namespace not in tree, node.prev)
				in_nodes.update(inp)


		for o in out_nodes:
			data_out += prod(o.output_shape)

		for i in in_nodes:
			if i.type == NODE_TYPES["SPLIT"]:
				data_in += prod(i.output_shape) * i.data["ways"]
			else:
				data_in += prod(i.output_shape)

		return {"data-in": data_in, "data-resident": data_resident, "data-out": data_out, "in_nodes": in_nodes, "out_nodes": out_nodes, "resident": resident}

	def find_namespaces(self, query: str) -> List[GraphNamespace]:
		""" returns namespaces matching query """
		names = list(filter(lambda ns: query in repr(ns), self.namespaces.keys()))
		return [self.namespaces[name] for name in names]


	def generate_depgraph(self):
		""" generates a simplified representation of the graph, realising data dependencies and compute clusters. """
		pass

	def export_graph(self, filename):
		""" export graph to a yaml file """
		pass
	
def import_graph(self, filename) -> CompGraph:
	""" import graph from a yaml file"""
	pass

def joined(predecessor: CompGraph, successor: CompGraph, pivot: GraphNode) -> CompGraph:
	""" joins two graphs together at a pivot node """
	pass

def bert_encoder(first = True, namespace = None) -> (GraphNode, GraphNode):
	# TODO: write parser from torch model to generalise
	""" returns a GraphNode hierarchy representing one encoder in the BERT model """
	dim = 1024
	hidden_dim = 4096
	len = 512
	heads = 16
	dh = dim//heads

	ffn_namespace = GraphNamespace("ffn", namespace)
	ffn_unit_namespaces = [GraphNamespace(f"FFN{i+1}", ffn_namespace) for i in range(len)]

	attention_namespace = GraphNamespace("self-attention", namespace)
	attention_head_namespaces = [GraphNamespace(f"head-{i+1}", attention_namespace) for i in range(heads)]
	KQV_namespaces = [GraphNamespace(f"{'QKV'[i%3]}", attention_head_namespaces[i//3]) for i in range(heads * 3)]
	

	end = GraphNode(n_type = NODE_TYPES["SCALE"], name = "Encoder Output (Residual Add)", namespace = namespace,prev = [GraphNode(n_type = NODE_TYPES["COMBINE"], data = {"axis": "row"}, name = "Post FFN Combine")])
	pre_FFN = GraphNode(n_type = NODE_TYPES["SPLIT"], name = "Pre FFN Split", namespace = namespace, data = {"axis": "row", "ways": len}, prev = [])
	FFN_wh = GraphNode(n_type = NODE_TYPES["DATA"], name = "Hidden Weights", namespace = ffn_namespace, output_shape = (dim, hidden_dim))
	FFN_wo = GraphNode(n_type = NODE_TYPES["DATA"], name = "Output Weights", namespace = ffn_namespace, output_shape = (hidden_dim, dim))

	FFNs = [
		GraphNode(n_type = NODE_TYPES["MATMUL"], name = "Output Layer", namespace = ffn_unit_namespaces[i], prev = [
			GraphNode(n_type = NODE_TYPES["MATMUL"], name = "Hidden Layer", namespace = ffn_unit_namespaces[i], prev = [
				pre_FFN,
				FFN_wh
			]),
			FFN_wo
		])
	for i in range(len)]

	end.prev[0].prev = FFNs

	pre_heads = GraphNode(n_type = NODE_TYPES["DATA" if first else "PASSTHROUGH"], namespace = namespace, output_shape = (len, dim), name = "Input Sequence")
	
	attention_heads = [
		GraphNode(n_type = NODE_TYPES["MATMUL"], name = "Output Layer", namespace = attention_head_namespaces[i], prev = [
			GraphNode(n_type = NODE_TYPES["SCALE"], name = "Softmax(*) + Scale", namespace = attention_head_namespaces[i], prev = [
				GraphNode(n_type = NODE_TYPES["MATMUL"], name = "KtQ", namespace = attention_head_namespaces[i], prev = [
					GraphNode(n_type = NODE_TYPES["MATMUL"], name = "Generator", namespace = KQV_namespaces[i * 3], prev = [
						pre_heads,
						GraphNode(n_type = NODE_TYPES["DATA"], name = "Weights", namespace = KQV_namespaces[i * 3], output_shape = (dim, dh))
					]),
					GraphNode(n_type = NODE_TYPES["MATMUL"], name = "Generator", namespace = KQV_namespaces[i * 3 + 1], prev = [
						pre_heads,
						GraphNode(n_type = NODE_TYPES["DATA"], name = "Weights", namespace = KQV_namespaces[i * 3 + 1], output_shape = (dim, dh))
					])
				], data = {"transpose_b": True})
			]),
			GraphNode(n_type = NODE_TYPES["MATMUL"], name = "Generator", namespace = KQV_namespaces[i * 3 + 2], prev = [
				pre_heads,
				GraphNode(n_type = NODE_TYPES["DATA"], name = "Weights", namespace = KQV_namespaces[i * 3 + 2], output_shape = (dim, dh))
			]),
		])
	for i in range(heads)]

	pre_FFN.prev = [
		GraphNode(n_type = NODE_TYPES["MATMUL"], name = "Output", namespace = attention_namespace, prev = [
			GraphNode(n_type = NODE_TYPES["COMBINE"], name = "Join Heads", namespace = attention_namespace, data = {"axis": "col"}, prev = attention_heads),
			GraphNode(n_type = NODE_TYPES["DATA"], name = "Output Weights", namespace = attention_namespace, output_shape = (dim, dim))
		])
	]

	start = pre_heads

	return (start, end)


if __name__ == "__main__":
	encoders = 24
	end = None
	cur = None
	for i in range(encoders, 0, -1):
		namespace = GraphNamespace(f"Encoder {i}", None)
		(encoder_start, encoder_end) = bert_encoder(i == 1, namespace)
		if cur is not None:
			cur.prev = [encoder_end]
		else:
			end = encoder_end

		cur = encoder_start

	graph = CompGraph(end)
	print(f"Total Costs (MAC): {graph.compute_costs()}")
	a = graph.find_namespaces("FFN512")[0]
	b = a.parent
	c = graph.namespaces["Encoder 24"]
	d = graph.namespaces["Encoder 24.self-attention.head-1"]
	e = graph.namespaces["Encoder 24.self-attention"]

	print(f"Costs (MAC) of a single FFN {repr(a)}: {graph.namespace_cost(a)}")
	print(f"Costs (MAC) of all FFN in an encoder {repr(b)}: {graph.namespace_cost(b)}")
	print(f"Direct Costs (MAC) of all FFN in an encoder {repr(b)}: {graph.direct_namespace_cost(b)}")
	print(f"Costs (MAC) of an encoder {repr(c)}: {graph.namespace_cost(c)}")
	print(f"Direct Costs (MAC) of an encoder {repr(c)}: {graph.direct_namespace_cost(c)}")
	print(f"Costs (MAC) of a single attention head {repr(d)}: {graph.namespace_cost(d)}")
	print(f"Direct Costs (MAC) of a single attention head {repr(d)}: {graph.direct_namespace_cost(d)}")
	print(f"Costs (MAC) of all attention heads in an encoder {repr(e)}: {graph.namespace_cost(e)}")
	print(f"Direct Costs (MAC) of all attention heads in an encoder {repr(e)}: {graph.direct_namespace_cost(e)}")