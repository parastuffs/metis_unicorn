import os
from natsort import natsorted
from sets import Set

def splitFile(rootDir, depth, ps):
	'''
	@return: None or list of dirs.
	The aim is to fetch the leaf directories of the recursivity tree.
	To do that, we return None when we reached the end of the tree (depth == 0).
	In the tree, we check is the returned value of the next branch is None.
	If it is, it means we are currently at the leaf and need to return the two
	sub-directories we created.
	If it's not, we are in the middle of the tree and we simply propagate the
	directories coming from the leaves.
	'''
	if depth > 0:

		os.chdir("/".join(ps.split('/')[:-1]))
		os.system("python " + phoneyScript + " -d " + rootDir + " -w 1")

		# Partition directory created by PHONEY.
		partDir = os.path.join(rootDir, natsorted(os.listdir(rootDir))[-1])
		# Need to create those for the progressive clustering.
		subPartDir0 = os.path.join(partDir, "progressive_cluster_" + str(depth) + "_0")
		subPartDir1 = os.path.join(partDir, "progressive_cluster_" + str(depth) + "_1")

		print("try to create new clustering dirs.")
		try:
			os.makedirs(subPartDir0)
			os.makedirs(subPartDir1)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

		print("New clustering dir created.")
		part0Clusters = Set()
		part1Clusters = Set()

		part0Instances = Set()
		part1Instances = Set()

		# First, get all instances by partition from the partition file.
		partitionFile = os.path.join(partDir, "metis_01_NoWires_area.hgr.part")
		with open(partitionFile, 'r') as f:
			line = f.readline().strip()
			while line:

				# line: <cluster name> <partition number>
				if line.split(' ')[1] == '0':
					part0Instances.add(line.split(' ')[0])
				elif line.split(' ')[1] == '1':
					part1Instances.add(line.split(' ')[0])
				else:
					print("Error, did not recognize partition '%s'", line.split(' ')[0])
				line = f.readline().strip()


		# From the instance per cluster Set, get all the clusters per partition.
		# Build the new output string at the same time.
		part0ClustInst = ""
		part1ClustInst = ""
		with open(os.path.join(rootDir,"ClustersInstances.out"), 'r') as f:
			line = f.readline().strip()
			while line:
				# print line
				cluster= line.split(' ')[0]
				instance = line.split(' ')[1]
				if instance in part0Instances:
					part0Clusters.add(cluster)
					part0ClustInst += line + "\n"
				elif instance in part1Instances:
					part1Clusters.add(cluster)
					part1ClustInst += line + "\n"
				else:
					print("ClustersInstances: Error, did not find instance '%s' in partition sets.", instance)
				line = f.readline().strip()


		#
		part0Nets = Set()
		part1Nets = Set()
		part0InstPerNet = dict()
		part1InstPerNet = dict()
		with open(os.path.join(rootDir,"InstancesPerNet.out"), 'r') as f:
			line = f.readline().strip()
			while line:
				# line: <net> <instance 1> <...> <instance n>

				net = line.split(' ')[0]
				# For each net, create a dictionary entry in both partitions.
				# We'll check later if we need to remove it.
				part0InstPerNet[net] = []
				part1InstPerNet[net] = []
				for instance in line.split(' ')[1:]:
					if instance in part0Instances:
						part0Nets.add(net)
						part0InstPerNet[net].append(instance)
					elif instance in part1Instances:
						part1Nets.add(net)
						part1InstPerNet[net].append(instance)
					else:
						print("InstancesPerNet: Error, did not find instance '%s' in partition sets.", instance)
				line = f.readline().strip()


		# Remove empty entries from the dictionary.
		keys = part0InstPerNet.keys()
		for k in keys:
			if len(part0InstPerNet[k]) < 1:
				del part0InstPerNet[k]
		keys = part1InstPerNet.keys()
		for k in keys:
			if len(part1InstPerNet[k]) < 1:
				del part1InstPerNet[k]


		# Write the new net files
		s = ""
		for net in part0Nets:
			s += net + "\n"
		with open(os.path.join(subPartDir0,"Nets.out"), 'w') as f:
			f.write(s)
		s = ""
		for net in part1Nets:
			s += net + "\n"
		with open(os.path.join(subPartDir1,"Nets.out"), 'w') as f:
			f.write(s)

		s0 = "NET  NUM_PINS  LENGTH"
		s1 = "NET  NUM_PINS  LENGTH"
		with open(os.path.join(rootDir,"WLnets.out"), 'r') as f:
			line = f.readline().strip()
			# Skip the first line.
			line = f.readline().strip()
			while line:
				# line: <net> <pins> <length>
				net = line.split(' ')[0]
				if net in part0Nets:
					s0 += line + "\n"
				elif net in part1Nets:
					s1 += line + "\n"
				# else:
					# print("Error, could not find net '%s' in partition sets.", net)
					# Empty nets can be silently ignored. 
				line = f.readline().strip()

		with open(os.path.join(subPartDir0,"WLnets.out"), 'w') as f:
			f.write(s0)
		with open(os.path.join(subPartDir1,"WLnets.out"), 'w') as f:
			f.write(s1)

		s = ""
		for k in part0InstPerNet:
			s += k
			for instance in part0InstPerNet[k]:
				s += " " + instance
			s += "\n"
		with open(os.path.join(subPartDir0,"InstancesPerNet.out"), 'w') as f:
			f.write(s)
		s = ""
		for k in part1InstPerNet:
			s += k
			for instance in part1InstPerNet[k]:
				s += " " + instance
			s += "\n"
		with open(os.path.join(subPartDir1,"InstancesPerNet.out"), 'w') as f:
			f.write(s)



		# Write the new cluster files
		s = ""
		for i in part0Clusters:
			s += str(i) + "\n"
		with open(os.path.join(subPartDir0,"Clusters.out"), 'w') as f:
			f.write(s)
		s = ""
		for i in part1Clusters:
			s += str(i) + "\n"
		with open(os.path.join(subPartDir1,"Clusters.out"), 'w') as f:
			f.write(s)


		# From the cluster per partition, build the str for ClustersArea.out.
		part0ClustAreaStr = "Name Type InstCount Boundary Area\n"
		part1ClustAreaStr = "Name Type InstCount Boundary Area\n"
		with open(os.path.join(rootDir,"ClustersArea.out"), 'r') as f:
			line = f.readline().strip()
			# skip first line.
			line = f.readline().strip()
			while line:

				clusterID = line.split(' ')[0]
				if clusterID in part0Clusters:
					part0ClustAreaStr += line + "\n"
				elif clusterID in part1Clusters:
					part1ClustAreaStr += line + "\n"
				else:
					print("Error, did not find cluster in partition sets.")
				line = f.readline().strip()

		with open(os.path.join(subPartDir0,"ClustersArea.out"), 'w') as f:
			f.write(part0ClustAreaStr)
		with open(os.path.join(subPartDir1,"ClustersArea.out"), 'w') as f:
			f.write(part1ClustAreaStr)

		with open(os.path.join(subPartDir0,"ClustersInstances.out"), 'w') as f:
			f.write(part0ClustInst)
		with open(os.path.join(subPartDir1,"ClustersInstances.out"), 'w') as f:
			f.write(part1ClustInst)


		# Recursivity \o/
		leafDirs0 = splitFile(subPartDir0, depth-1, ps)
		leafDirs1 = splitFile(subPartDir1, depth-1, ps)

		if leafDirs0 is None and leafDirs1 is None:
			return [subPartDir0, subPartDir1]
		else:
			for leaf in leafDirs1:
				leafDirs0.append(leaf)
			return leafDirs0
	else:
		return None

if __name__ == "__main__":

	phoneyScript = "/home/para/dev/metis_unicorn/script/phoney.py"

	depth = 3
	rootDir = "/home/para/dev/metis_unicorn/temp_design/ldpc_random_0"

	leafDirs = splitFile(rootDir, depth, phoneyScript)
	print leafDirs
	print("Total nodes: ", len(leafDirs))