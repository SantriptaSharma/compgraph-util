from graph import *

if __name__ == "__main__":
    ns = GraphNamespace("encoder")
    g = CompGraph(bert_encoder(True, ns)[1])
    head = g.namespace_data(g.find_namespaces("head")[0])
    attention = g.namespace_data(g.find_namespaces("self-attention")[0])
    ffn = g.namespace_data(g.find_namespaces("ffn")[0])
    encoder = g.namespace_data(g.find_namespaces("encoder")[0])