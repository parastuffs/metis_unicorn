import os
from natsort import natsorted
from sets import Set
import shutil
import sys, getopt
import errno

def timeSorted(path):
	# https://stackoverflow.com/a/4500607/3973030
	mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
	return list(sorted(os.listdir(path), key=mtime))

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
		partDir = os.path.join(rootDir, timeSorted(rootDir)[-1])
		# Need to create those for the progressive clustering.
		subPartDir0 = os.path.join(partDir, "progressive_cluster_" + str(depth) + "_0")
		subPartDir1 = os.path.join(partDir, "progressive_cluster_" + str(depth) + "_1")

		try:
			os.makedirs(subPartDir0)
			os.makedirs(subPartDir1)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

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
		filepath = os.path.join(rootDir,"InstancesPerNet.out")
		try:
			with open(filepath, 'r') as f:
				lines = f.read().splitlines()
		except IOError:
			with open(os.sep.join([os.sep.join(filepath.split(os.sep)[:-2]), filepath.split(os.sep)[-1]]), 'r') as f:
				lines = f.read().splitlines()
		finally:
			for line in lines:
				line = line.strip()
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
		filepath = os.path.join(subPartDir0,"Nets.out")
		try:
			with open(filepath, 'w') as f:
				f.write(s)
		except IOError:
			with open(os.sep.join([os.sep.join(filepath.split(os.sep)[:-2]), filepath.split(os.sep)[-1]]), 'w') as f:
				f.write(s)
		s = ""
		for net in part1Nets:
			s += net + "\n"
		filepath = os.path.join(subPartDir1,"Nets.out")
		try:
			with open(filepath, 'w') as f:
				f.write(s)
		except IOError:
			with open(os.sep.join([os.sep.join(filepath.split(os.sep)[:-2]), filepath.split(os.sep)[-1]]), 'w') as f:
				f.write(s)

		s0 = "NET  NUM_PINS  LENGTH\n"
		s1 = "NET  NUM_PINS  LENGTH\n"
		filepath = os.path.join(rootDir,"WLnets.out")
		try:
			with open(filepath, 'r') as f:
				lines = f.read().splitlines()
		except IOError:
			with open(os.sep.join([os.sep.join(filepath.split(os.sep)[:-2]), filepath.split(os.sep)[-1]]), 'r') as f:
				lines = f.read().splitlines()
		finally:
			# Skip the first line.
			for line in lines[1:]:
				line = line.strip()
				# line: <net> <pins> <length>
				net = line.split(' ')[0]
				if net in part0Nets:
					s0 += line + "\n"
				elif net in part1Nets:
					s1 += line + "\n"
				# else:
					# print("Error, could not find net '%s' in partition sets.", net)
					# Empty nets can be silently ignored.

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



def recycleTree(rootDir, depth):
	'''
	@rootDir: full os path
	@return: None or rootDir in an array
	'''
	if depth > 0:
		finalDirs = []
		isPartDir = False
		isClustDir = False
		if "partitions" in rootDir.split(os.sep)[-1]:
			isPartDir = True
		elif "progressive_cluster" in rootDir.split(os.sep)[-1]:
			isClustDir = True
		else:
			return None

		if isPartDir:
			depth -= 1

		for d in os.listdir(rootDir):
			subdir = os.path.join(rootDir, d)
			if os.path.isdir(subdir):
				returnDirs = recycleTree(subdir, depth)
				for sub in returnDirs:
					if sub is not None:
						finalDirs.append(sub)

		return finalDirs
	else:
		return [rootDir]






def mergeClusters(dirs, rootDir):
	'''
	This function should first create a new directory for the new clusters.
	It should be placed one level above the root directory, so that it lies next to
	the original clustering directory.

	@rootDir: os path
	Its last directory should be structured as <design>_<clustering method>_<clustering size>.
	'''

	progClustDir = os.path.join((os.sep).join(rootDir.split(os.sep)[:-1]),(rootDir.split(os.sep)[-1]).split('_')[0] + "_progressive_" + str(len(dirs)))
	# print progClustDir

	try:
		os.makedirs(progClustDir)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	# First copy the unchanged files.
	# shutil.copy(os.path.join(rootDir, "CellCoord.out"), progClustDir)
	# shutil.copy(os.path.join(rootDir, "InstancesPerNet.out"), progClustDir)
	# shutil.copy(os.path.join(rootDir, "Nets.out"), progClustDir)
	# shutil.copy(os.path.join(rootDir, "WLnets.out"), progClustDir)

	clustersInstances = ""
	clustersArea = [0] * len(dirs)
	clustersInstancesCount = [0] * len(dirs)
	clustersAreaStr = "Name Type InstCount Boundary Area\n"
	clustersStr = ""
	for i in range(0, len(dirs)):
		d = dirs[i]
		with open(os.path.join(d, "ClustersArea.out"), 'r') as f:

			line = f.readline().strip()
			# Skip header line
			line = f.readline().strip()
			while line:
				clustersArea[i] += float(line.split(' ')[-1])
				clustersInstancesCount[i] += int(line.split(' ')[2])


				line = f.readline().strip()
			clustersAreaStr += str(i) + " exclusive " + str(clustersInstancesCount[i]) + " (0,0) (0,0) " + str(clustersArea[i]) + "\n"

		with open(os.path.join(d, "ClustersInstances.out"), 'r') as f:
			clustersInstances += str(i)
			line = f.readline().strip()
			while line:
				clustersInstances += " " + line.split(' ')[-1]
				line = f.readline().strip()
			clustersInstances += "\n"
		clustersStr += str(i) + "\n"


	with open(os.path.join(progClustDir, "ClustersArea.out"), 'w') as f:
		f.write(clustersAreaStr)
	with open(os.path.join(progClustDir, "ClustersInstances.out"), 'w') as f:
		f.write(clustersInstances)
	with open(os.path.join(progClustDir, "Clusters.out"), 'w') as f:
		f.write(clustersStr)






if __name__ == "__main__":

	phoneyScript = ""
	rootDir = ""
	depth = None
	# Reuse the existing folders.
	# Warning: the input directory needs to be a 'partitions_...' folder.
	reuse = False

	try:
		opts, args = getopt.getopt(sys.argv[1:],"d:s:n:r")
	except getopt.GetoptError:
		print "YO FAIL"
	else:
		for opt, arg in opts:
			if opt == "-d":
				rootDir = arg
			if opt == "-s":
				phoneyScript = arg
			if opt == "-n":
				depth = int(arg)
			if opt == "-r":
				reuse = True
		if rootDir == "":
			# rootDir = "/home/para/dev/def_parser/2018-03-14_17-00-18/ldpc-4x4-serial_random_0"
			rootDir = "/home/para/dev/def_parser/2019-03-07_11-41-34_ldpc_OneToOne/ldpc_progressive-wl_0"
		if phoneyScript == "":
			phoneyScript = "/home/para/dev/metis_unicorn/script/phoney.py"
		if depth is None:
			depth = 3

	if not reuse:
		leafDirs = splitFile(rootDir, depth, phoneyScript)
	else:
		leafDirs = recycleTree(rootDir, depth)
		# In case of recycling, we get an extra folder as input.
		rootDir = os.sep.join(rootDir.split(os.sep)[:-1])
	# print leafDirs
	print("Total nodes: ", len(leafDirs))
	mergeClusters(leafDirs, rootDir)