import os
import getopt, sys

class Graph:
	def __init__(self):
		self.edges = dict()
		self.vertices = dict()

	def addEdge(self, e):
		self.edges[e.id] = e

	def addVertex(self, v):
		self.vertices[v.id] = v


class Vertex:
	def __init__(self, id):
		self.id = id
		self.edges = [] # Array of Edge objects
		self.weight = 1

	def addEdge(self, e):
		self.edges.append(e)


class Edge:
	def __init__(self, id, w=1):
		self.id = id
		self.vertices = [] # Array of Vertex objects
		self.weight = w

	def addVertex(self, v):
		self.vertices.append(v)

	def setWeight(self, w):
		self.weight = w



if __name__ == "__main__":

	circutIn = ""

	try:
		opts, args = getopt.getopt(sys.argv[1:],"d:")
	except getopt.GetoptError:
		print("Something went wrong with the arguments.")
	else:
		for opt, arg in opts:
			if opt == "-d":
				circutIn = arg

	graph = Graph()

	if circutIn == "":
		circutIn = "/home/para/dev/circut10612/circut_v1.0612/tests/G1"

	metisIn = circutIn.split(os.sep)[-1] + "_metis"

	with open(circutIn, 'r') as f:
		line = f.readline().strip()
		totVert = int(line.split(' ')[0])
		totEdg = int(line.split(' ')[1])
		for i in range(1, totVert + 1):
			v = Vertex(i)
			graph.addVertex(v)
		for i in range(1, totEdg + 1):
			e = Edge(i)
			graph.addEdge(e)

		line = f.readline().strip()
		i = 1
		while line:
			# get the index of concerned vertices
			vi = int(line.split(' ')[0])
			vj = int(line.split(' ')[1])
			wij = int(line.split(' ')[2])
			# Set those in the current edge
			graph.edges[i].addVertex(graph.vertices[vi])
			graph.edges[i].addVertex(graph.vertices[vj])
			graph.edges[i].setWeight(wij)
			# Add a reference to the edge inside the connected vertices
			graph.vertices[vi].addEdge(graph.edges[i])
			graph.vertices[vj].addEdge(graph.edges[i])
			i += 1
			line = f.readline().strip()

	s = str(len(graph.vertices)) + " " + str(len(graph.edges)) + " 001\n"
	for key in graph.vertices:
		v = graph.vertices[key]
		vi = v.id
		for e in v.edges:
			vj = e.vertices[0].id
			if vi == vj:
				vj = e.vertices[1].id
			if vi == vj:
				print("vi == vj, you're screwed boi.")
			s += str(vj) + " " + str(e.weight) + " "
		s += "\n"

	with open(metisIn, 'w') as f:
		f.write(s)