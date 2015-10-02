"""
PHONEY [Pronounce 'poney'], Partitioning of Hypergraph Obviously Not EasY

Want to use it? Then check the 'HMETIS_PATH' and 'dirs' variables.
"""

"""
TODO
> Doc
> Change the ID of the clusters so that it begins at 1 instead of 0.
"""

import math
from subprocess import call
import copy
import matplotlib.pyplot as plt
import threading
from multiprocessing import Process, Pipe

global HMETIS_PATH
HMETIS_PATH = "/home/para/dev/metis_unicorn/hmetis-1.5-linux/"
METIS_PATH = "/home/para/dev/metis_unicorn/metis/bin/"
EDGE_WEIGHTS_TYPES = 7
VERTEX_WEIGHTS_TYPES = 1
WEIGHT_COMBILI_COEF = 0.5
MAX_WEIGHT = 1000
SIMPLE_GRAPH = False
THREADS = 3
CLUSTER_INPUT_TYPE = 0  # 0: standard .out lists
                        # 1: Ken's output

def printProgression(current, max):
    progression = ""
    if current == max/10:
        progression = "10%"
    elif current == max/5:
        progression = "20%"
    elif current == 3*max/10:
        progression = "30%"
    elif current == 2*max/5:
        progression = "40%"
    elif current == max/2:
        progression = "50%"
    elif current == 3*max/5:
        progression = "60%"
    elif current == 7*max/10:
        progression = "70%"
    elif current == 4*max/5:
        progression = "80%"
    elif current == 9*max/10:
        progression = "90%"
    elif current == max:
        progression = "100%"
    return progression


def buildHyperedges(processID, startIndex, endIndex, nets, clusters, pipe):

    hyperedges = []
    print "in process"
    # print clusters
    i = startIndex
    while i < endIndex:
        net = nets[i]
        connectedClusters = list()

        # Now, for each net, we get its list of instances.
        for netInstance in net.instances: # netInstance are Instance object.

            # And for each instance in the net, we check to which cluster
            # that particular instance belongs.
            for cluster in clusters:
                # Try to find the instance from the net in the cluster
                if cluster.searchInstance(netInstance):
                    # If found, see if the cluster has already been added
                    # to connectedClusters.
                    j = 0
                    clusterFound = False
                    while j < len(connectedClusters) and not clusterFound:
                        if connectedClusters[j].ID == cluster.ID:
                            clusterFound = True
                        else:
                            j += 1
                    if not clusterFound:
                        connectedClusters.append(cluster)
 
        # Append the list A of connected clusters to the list B of hyperedges
        # only if there are more than one cluster in list A.
        if len(connectedClusters) > 1:
            if SIMPLE_GRAPH:
                for i in xrange(0,len(connectedClusters)):
                    for j in xrange(i + 1, len(connectedClusters)):
                        hyperedge = Hyperedge()
                        hyperedge.addCluster(connectedClusters[i])
                        hyperedge.addCluster(connectedClusters[j])
                        hyperedge.addNet(net)
                        hyperedges.append(hyperedge)
            else:
                hyperedge = Hyperedge()
                for cluster in connectedClusters:
                    # print cluster
                    hyperedge.addCluster(cluster)
                hyperedge.addNet(net)
                hyperedges.append(hyperedge)

        progression = printProgression(i - startIndex, endIndex - startIndex)
        if progression != "":
            print "Process " + str(processID) + " " + progression
        i += 1


    pipe.send(hyperedges)
    pipe.close()


class Graph():
    def __init__(self):
        self.clusters = [] # list of Cluster objects
        self.nets = [] # list of Net objects.
        self.hyperedges = []
        self.hyperedgeWeightsMax = []   # Maximum weight for each weight type.
                                        # Ordered by weight type.
        self.logfilename = "graph.log"
        call(["rm","-rf","graph.log"])

    def WriteLog(self, obj):
        f = open(self.logfilename, "a")
        f.write(str(obj)+"\n")
        f.close()

    def findClusterByName(self, clusterName):
        found = False
        clusterID = 0
        while not found and clusterID < len(self.clusters):
            if self.clusters[clusterID].name == clusterName:
                found = True
            else:
                clusterID += 1
        return found, clusterID

    def ReadClusters(self, filename, hrows, frows):
        print (str("Reading clusters file: " + filename))
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        # Remove the header lines
        for i in xrange(0, hrows):
            del lines[0]
        # Remove the footer lines
        for i in xrange(0,frows):
            del lines[-1]

        for i, line in enumerate(lines):
            line = line.strip(' \n')
            clusterDataRow = line.split()
            cluster = Cluster(clusterDataRow[0], i, False)
            self.clusters.append(cluster)

            lowerBounds = clusterDataRow[3][1:-1].split(",")    # Exclude the first and last
                                                                # chars: '(' and ')'.
            upperBounds = clusterDataRow[4][1:-1].split(",")
            self.clusters[i].setBoundaries(float(lowerBounds[0]), float(lowerBounds[1]),
                float(upperBounds[0]), float(upperBounds[1]))
            
            self.clusters[i].setArea(float(clusterDataRow[5]))


    def readClustersInstances(self, filename, hrows, frows):
        # print "--------------------------------------------------->"
        print (str("Reading clusters instances file: " + filename))
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        if CLUSTER_INPUT_TYPE == 0:
            # Remove the header lines
            for i in xrange(0, hrows):
                del lines[0]
            # Remove the footer lines
            for i in xrange(0,frows):
                del lines[-1]

            for line in lines:
                line = line.strip(' \n')
                line = line.replace('{','')
                line = line.replace('}','')
                # print line
                clusterInstancesRow = line.split()
                found, clusterID = self.findClusterByName(clusterInstancesRow[0])
                if found:
                    del clusterInstancesRow[0]
                    for i, instanceName in enumerate(clusterInstancesRow):
                        # instance = Instance(instanceName)
                        # self.clusters[clusterID].addInstance(instance)
                        self.clusters[clusterID].addInstance(instanceName)
        elif CLUSTER_INPUT_TYPE == 1:
            for line in lines:
                line = line.strip(' \n')
                clusterInstancesRow = line.split()
                found, clusterID = self.findClusterByName(clusterInstancesRow[3])
                if found:
                    self.clusters[clusterID].addInstance(clusterInstancesRow[2])



    def readMemoryBlocks(self, filename, hrows, frows):
        print (str("Reading memory blocks file: " + filename))
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        # Remove the header lines
        for i in xrange(0, hrows):
            del lines[0]
        # Remove the footer lines
        for i in xrange(0,frows):
            del lines[-1]

        for i in xrange(0, len(lines)):
            lines[i] = " ".join(lines[i].split())

        i = 0
        existingClustersCount = len(self.clusters) # Clusters already in the list before.
        blockCount = 0
        while i < len(lines):
            line = lines[i]
            #[0]: block name
            #[2]: instance count
            #[6]: total area
            #[10]: instance
            blockRow = line.split()
            # print blockRow
            if len(line) > 0:
                memoryBlock = Cluster(blockRow[10], existingClustersCount + blockCount, True)
                blockCount += 1
                moduleArea = float(blockRow[6])
                memoryBlock.setArea(moduleArea)
                memoryBlock.addInstance(blockRow[10])
                self.clusters.append(memoryBlock)

                for j in xrange(1, int(blockRow[2])):
                    i += 1
                    line = lines[i]
                    subBlockRow = line.split()
                    memoryBlock = Cluster(subBlockRow[1], existingClustersCount + blockCount, True)
                    blockCount += 1
                    memoryBlock.setArea(moduleArea)
                    memoryBlock.addInstance(subBlockRow[1])
                    self.clusters.append(memoryBlock)

            i += 1


    def readNetsWireLength(self, filename, hrows, frows):
        print (str("Reading wire length nets file: " + filename))
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        # Remove the header lines
        for i in xrange(0, hrows):
            del lines[0]
        # Remove the footer lines
        for i in xrange(0,frows):
            del lines[-1]

        for i in xrange(0, len(lines)):
            lines[i] = " ".join(lines[i].split())
        lines.sort()

        for i, line in enumerate(lines):
            netDataRow = line.split()

            net = Net(netDataRow[0], i)
            net.setPinAmount(int(netDataRow[1]))
            net.setWL(int(float(netDataRow[2])) + 1)    # + 1 here to make sure 
                                                        #we don't use a net with WL = 0
            self.nets.append(net)
            progression = printProgression(i, len(lines))
            if progression != "":
                print progression


    def readNets(self, filename, hrows, frows):
        print (str("Reading nets file: " + filename))
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        # Remove the header lines
        for i in xrange(0, hrows):
            del lines[0]
        # Remove the footer lines
        for i in xrange(0,frows):
            del lines[-1]

        lines.sort()

        # As the nets list is sorted, we should not run through all the
        # self.nets list for each line.
        # For each line, move forward in self.nets. If the net in the line
        # is found, great! If not, no biggie, just keep moving forward.
        netID = 0
        i = 0
        while i < len(lines) and netID < len(self.nets):
            line = lines[i]
            line = line.strip(' \n')
            line = line.replace('{','')
            line = line.replace('}','')
            netDataRow = line.split()
            if self.nets[netID].name == netDataRow[0]:
                del netDataRow[0]
                for instanceName in netDataRow:
                    # instance = Instance(instanceName)
                    # self.nets[netID].addInstance(instance)
                    self.nets[netID].addInstance(instanceName)
                netID += 1
            i += 1

            progression = printProgression(i, len(lines))
            if progression != "":
                print progression

    
    def findHyperedges(self):
        print "Building hyperedges"

        processes = []
        pipes = []
        print "Before processes"
        # print self.clusters

        for i in xrange(0, THREADS):
            parent_pipe, child_pipe = Pipe()
            pipes.append(parent_pipe)
            process = Process(target=buildHyperedges, args=(i, \
                i * len(self.nets) / THREADS, \
                (i + 1) * len(self.nets) / THREADS, \
                self.nets, \
                self.clusters, \
                child_pipe,))
            process.start()
            processes.append(process)

        for i, pipe in enumerate(pipes):
            print "Waiting pipe from process " + str(i)
            hyperedges = pipe.recv()
            for hyperedge in hyperedges:
                self.hyperedges.append(hyperedge)

        for process in processes:
            process.join()




        s = ""
        for hyperedge in self.hyperedges:
            s += str(len(hyperedge.clusters)) + "\t" + str(hyperedge.nets[0].name)
            for cluster in hyperedge.clusters:
                s += " " + str(cluster.name)
            s += "\n"
        with open("raw_hyperedges.out", 'w') as f:
            f.write(s)


        print "Merging hyperedges"
        i = 0
        while i < len(self.hyperedges):
            hyperedgeA = self.hyperedges[i]
            duplicate = False
            clusterAMerged = False  # Flag set in case an hyperedgeA is merged
                                    # into an hyperedgeB.
            j = i + 1
            while j < len(self.hyperedges) and not clusterAMerged:
                hyperedgeB = self.hyperedges[j]
                if len(hyperedgeA.clusters) == len(hyperedgeB.clusters):
                    duplicate = True
                    k = 0
                    # Check if all clusters in the hyperedge are the same
                    while k < len(hyperedgeA.clusters) and duplicate:
                        clusterA = hyperedgeA.clusters[k]
                        clusterB = hyperedgeB.clusters[k]
                        if clusterA.name != clusterB.name:
                            duplicate = False
                        else:
                            k += 1
                    if duplicate:
                        # Append the net from hyperedgeB to hyperedgeA.
                        # At this point, hyperedgeB only has one edge.
                        hyperedgeA.addNet(hyperedgeB.nets[0])
                        hyperedgeA.connectivity += 1
                        del self.hyperedges[j]
                    else:
                        j += 1

                else:
                    j += 1

            # If hyperedgeA has not been merged, inspect the next one.
            # Otherwise, hyperedgeA has been deleted and all the following
            # elements have been shifted, thus no need to increment the index.
            if not clusterAMerged:
                i += 1

            progression = printProgression(i, len(self.hyperedges))
            if progression != "":
                print progression

        if SIMPLE_GRAPH:
            print "Prepare simple graph"
            for i, hyperedge in enumerate(self.hyperedges):
                for k, clusterA in enumerate(hyperedge.clusters):
                    for l, clusterB in enumerate(hyperedge.clusters):
                        if l != k:
                            clusterA.connectedClusters.append(clusterB)
                            clusterA.connectedEdges.append(hyperedge)




    def generateMetisInput(self, filename, edgeWeightType, vertexWeightType):
        print "Generating METIS input file..."
        s = ""
        if SIMPLE_GRAPH:
            s = str(len(self.clusters)) + " " + str(len(self.hyperedges)) + " 011"

            for i, cluster in enumerate(self.clusters):
                s += "\n" + str(cluster.weightsNormalized[vertexWeightType])
                for j in xrange(0, len(cluster.connectedClusters)):
                    s += " " + str(cluster.connectedClusters[j].ID + 1) + \
                        " " + str(cluster.connectedEdges[j].weightsNormalized[edgeWeightType])


        else:
            s = str(len(self.hyperedges)) + " " + str(len(self.clusters)) + " 11"
            for i, hyperedge in enumerate(self.hyperedges):
                s += "\n" + str(hyperedge.weightsNormalized[edgeWeightType]) + " "
                for cluster in hyperedge.clusters:
                    s += str(cluster.ID + 1) + " "  # hmetis does not like to have hyperedges
                                                    # beginning with a cluster of ID '0'.
            for cluster in self.clusters:
                s += "\n" + str(cluster.weightsNormalized[vertexWeightType])
        with open(filename, 'w') as file:
            file.write(s)


    def computeHyperedgeWeights(self, normalized):
        print "Generating weights of hyperedges."

        self.hyperedgeWeightsMax = [0] * EDGE_WEIGHTS_TYPES
        for weightType in xrange(0, EDGE_WEIGHTS_TYPES):
            for i, hyperedge in enumerate(self.hyperedges):
                weight = 0
                if weightType == 0:
                    # Number of wires
                    weight = hyperedge.connectivity
                elif weightType == 1:
                    # Wire length
                    wlTot = 0
                    for net in hyperedge.nets:
                        wlTot += net.wl
                    wlAvg = wlTot / len(hyperedge.nets)
                    weight = wlAvg
                elif weightType == 2:
                    # 1/#wires
                    weight = 1.0 / hyperedge.weights[0]
                elif weightType == 3:
                    # 1/Wire length
                    weight = 1.0 / hyperedge.weights[1]
                elif weightType == 4:
                    # wire length and number of wires
                    weight = \
                        WEIGHT_COMBILI_COEF * hyperedge.weightsNormalized[0] + \
                        (1 - WEIGHT_COMBILI_COEF) * hyperedge.weightsNormalized[1]
                elif weightType == 5:
                    # 1 / (wire length and number of wires)
                    weight = 1.0 / ( \
                        WEIGHT_COMBILI_COEF * hyperedge.weightsNormalized[0] + \
                        (1 - WEIGHT_COMBILI_COEF) * hyperedge.weightsNormalized[1] )
                elif weightType == 6:
                    # Constant
                    weight = 1

                hyperedge.setWeight(weightType, weight)
                # Save the max
                if weight > self.hyperedgeWeightsMax[weightType]:
                    self.hyperedgeWeightsMax[weightType] = weight

            # Normalize
            for hyperedge in self.hyperedges:
                hyperedge.setWeightNormalized(weightType, int(((hyperedge.weights[weightType] * MAX_WEIGHT) / self.hyperedgeWeightsMax[weightType])) + 1)
                



    def computeVertexWeights(self):
        print "Generating weights of vertex."
        self.clusterWeightsMax = [0] * VERTEX_WEIGHTS_TYPES

        for cluster in self.clusters:
            for weightType in xrange(0, VERTEX_WEIGHTS_TYPES):
                if weightType == 0:
                    weight = cluster.area
                elif weightType == 1:
                    weight = 1

                if weight > self.clusterWeightsMax[weightType]:
                    self.clusterWeightsMax[weightType] = weight

                cluster.setWeight(weightType, weight)

        for cluster in self.clusters:
            for weightType in xrange(0, VERTEX_WEIGHTS_TYPES):
                weight = ((cluster.weights[weightType] * MAX_WEIGHT) / self.clusterWeightsMax[weightType]) + 1
                cluster.setWeightNormalized(weightType, int(weight))



    def GraphPartition(self, filename):
        print "--------------------------------------------------->"
        # print "Running partition on cost: ", CostFunction
        print "Running hmetis with " + filename
        # call(["/Users/drago/bin/hmetis-1.5-osx-i686/hmetis",filename,"2","5","20","1","1","1","0","0"])
        # hmetis graphFile Nparts UBfactor Nruns Ctype Rtype Vcycle Reconst dbglvl
        command = ""
        if SIMPLE_GRAPH:
            command = METIS_PATH + "gpmetis " + filename + " 2 -dbglvl=0"
        else:
            command = HMETIS_PATH + "hmetis " + filename + " 2 1 20 1 1 1 0 8"
            print "Calling '" + command + "'"
        # call([HMETIS_PATH + "hmetis",filename,"2","5","20","1","1","1","0","0"])
        call(command.split())
    
    def WritePartitionDirectives(self, metisFileIn, metisFileOut):
        # print "--------------------------------------------------->"
        print "Write tcl file for file: ", metisFileIn
        try:
            fOut = open(metisFileOut, "w")
            fIn = open(metisFileIn, "r")
        except IOError as e:
            print "Can't open file"
            return False
        data = fIn.readlines()
        for i, cluster in enumerate(self.clusters):
            if cluster.blackbox:
                fOut.write(str('add_to_die -inst ' + "\t" + str(cluster.name) + \
                    "\t -die Die" + str(data[i])))
            else:
                fOut.write(str('add_to_die -cluster ' + "\t" + str(cluster.name) + \
                    "\t -die Die" + str(data[i])))
        fOut.close()
        fIn.close()
        print "Done!"
        # print "<---------------------------------------------------\n"
        
    def dumpClusters(self):
        s = ""
        for cluster in self.clusters:
            s += str(cluster.ID) + "\t" + cluster.name + "\n"
        with open("clusters", 'w') as f:
            f.write(s)

    def hammingReport(self, filenames):
        partitions = []
        table = ""

        part = 0
        for filename in filenames:
            # Print the header, but ommit '0'
            if part > 0:
                table += str(part)# Table header
            table += "\t"
            part += 1

            with open(filename, 'r') as f:
                lines = f.read().splitlines()
                for i in xrange(0, len(lines)):
                    lines[i] = lines[i].strip(' \n')
                partitions.append(lines)
                print lines

        table += "\n"

        for i in xrange(0, len(partitions) - 1): # ommit the last line, it has already been done as the last column
            table += str(i) + "\t" # Row name
            table += "\t" * i
            for j in xrange(i + 1, len(partitions)):
                hammingDistance = 0
                for bit in xrange(0, len(partitions[i])):
                    if partitions[i][bit] != partitions[j][bit]:
                        hammingDistance += 1
                # Hamming distance normalized, limited to two decimal points
                table += str("%.2f" % (float(hammingDistance)/len(partitions[i]))) + "\t"
            table += "\n"

        part = 0
        table += "Legend:\n"
        for filename in filenames:
            table += str(part) + ":\t" + filename + "\n"
            part += 1
        print table



    def plotWeights(self):
        # weights = [0] * EDGE_WEIGHTS_TYPES
        weights = []
        styles = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        for i in xrange(0, EDGE_WEIGHTS_TYPES):
            weights.append(list())
        for hyperedge in self.hyperedges:
            for i in xrange (0, EDGE_WEIGHTS_TYPES):
                weights[i].append(hyperedge.weightsNormalized[i])
        print weights[0]
        for i in xrange(0, EDGE_WEIGHTS_TYPES):
            plt.plot(weights[i], styles[i])
        plt.show()


    def computePartitionArea(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        for i in xrange(0, len(lines)):
            lines[i] = " ".join(lines[i].split())

        partitionsArea = [0] * 2

        for line in lines:
            lineData = line.split()
            found, index = self.findClusterByName(lineData[2])
            if lineData[4] == "Die0":
                partitionsArea[0] += self.clusters[index].area
            elif lineData[4] == "Die1":
                partitionsArea[1] += self.clusters[index].area

        print partitionsArea
              




class Net:
    def __init__(self, name, netID):
        self.name = name
        self.ID = netID
        self.wl = 0 # wire length
        self.instances = []
        self.pins = 0 # number of pins

    def setPinAmount(self, pins):
        self.pins = pins

    def setWL(self, wl):
        self.wl = wl

    def addInstance(self, instance):
        self.instances.append(instance)

    def searchInstance(self, instance):
        found = False
        try:
            self.instances.index(instance)
        except:
            pass
        else:
            found = True
        return found


class Cluster:
    def __init__(self, name, clusterID, blackbox):
        self.name = name
        self.ID = clusterID
        self.instances = []
        self.boudndaries = [[0, 0], [0 ,0]] # [[lower X, lower U], [upper X, upper Y]] (floats)
        self.area = 0 # float
        self.weights = []   # [0] = area
                            # [1] = power
                            # [2] = area & power
        self.weightsNormalized = []
        self.connectedClusters = [] # List of Cluster objects connected to this cluster.
        self.connectedEdges = []    # List of Hyperedge objects.
                                    # They are the edges connecting this cluster to others.
                                    # Those hyperedges are needed to establish weights.
                                    # connectedEdges[i] connects to conectedClusters[i]
        self.blackbox = blackbox    # Boolean
                                    # When a cluster is a blackbox, it's composed of only
                                    # one instance. Hence, when writting its directive into
                                    # the .tcl file, use the '-inst' modifier.
                                    # e.g. Memory are blackboxes.

        self.weights = [0] * VERTEX_WEIGHTS_TYPES
        self.weightsNormalized = [0] * VERTEX_WEIGHTS_TYPES

    def setBoundaries(self, lowerX, lowerY, upperX, upperY):
        self.boudndaries[0][0] = lowerX
        self.boudndaries[0][1] = lowerY
        self.boudndaries[1][0] = upperX
        self.boudndaries[1][1] = upperY

    def addInstance(self, instance):
        """
        instance: Instance object
        """
        self.instances.append(instance)

    def searchInstance(self, instance):
        # print "Searching " + instance.name + " in cluster " + self.name
        found = False
        # i = 0
        # while not found and i < len(self.instances):
        #     if self.instances[i].name == instance.name:
        #         found = True
        #     else:
        #         i += 1
        try:
            self.instances.index(instance)
        except:
            pass
        else:
            found = True
        return found

    def setArea(self, area):
        self.area = area

    def setWeight(self, index, weight):
        self.weights[index] = weight

    def setWeightNormalized(self, index, weight):
        self.weightsNormalized[index] = weight

    def searchConnectedCluster(self, cluster):
        found = False
        index = -1
        try:
            index = self.connectedClusters.index(cluster)
        except:
            pass
        else:
            found = True
        return index, found


class Hyperedge:
    def __init__(self):
        self.nets = [] # list of Nets
        self.clusters = [] # list of Clusters
        self.weights = []   # [0] = Number of wires
                            # [1] = Wire length
                            # [2] = 1/number of wires
                            # [3] = 1/wire length
                            # [4] = WEIGHT_COMBILI_COEF * number of wires +
                            #       (1 - WEIGHT_COMBILI_COEF) * wire length
                            # [5] = 1/[4]
        self.weightsNormalized = [] # Same weights, but normalized like so:
                                    # (weight / max_weight) * 1000

        self.weights = [0] * EDGE_WEIGHTS_TYPES
        self.weightsNormalized = [0] * EDGE_WEIGHTS_TYPES
        self.connectivity = 1 # Number of connection between the cluster inside the hyperedge.

    def addNet(self, net):
        self.nets.append(net)

    def addCluster(self, cluster):
        self.clusters.append(cluster)

    def setWeight(self, index, weight):
        self.weights[index] = weight

    def setWeightNormalized(self, index, weight):
        self.weightsNormalized[index] = weight


class Instance:
    def __init__(self, name):
        self.name = name

#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------
if __name__ == "__main__":
#==========================================================================
# Read rpt gen with report physical.connectivity -net_threshold 0
#------------------------------------------------------------------------------------------------------------------------------------------------
#SourceBlock | SourceType | SourceBlockArea | ConnectedBlock | ConnectedType | ConnectedBlockArea | E1->E2 | E2->E1 | E1-E2 | Total Connections
#------------------------------------------------------------------------------------------------------------------------------------------------
#   connectivity = RPTFile("4rpt.phys_conn.rpt", 34, 2)
#   connectivity.Read()
#==========================================================================
# Read rpt gen with report physical.hierarchy
#    -------------------------------------------- 
#    Name | Type | Area | Inst | Cnt |  Area(%) 
#    -------------------------------------------- 
    dirs=["../input_files/"]
    # dirs=["../ccx/"]
    # dirs = ["../MPSoC/"]
    # dirs = ["../spc_L3/"]
    # dirs = ["../spc_L2/"]

    graph = Graph()
#    RunCosts = [0, 1, 2]
#    RunCosts = [3, 4, 5]
    RunCosts = [0, 1, 2, 3, 4, 5]

    for mydir in dirs:
        clustersAreaFile = mydir + "ClustersArea.out"
        clustersInstancesFile = mydir + "ClustersInstances.out"
        netsInstances = mydir + "InstancesPerNet.out"
        netsWL = mydir + "WLnets.out"
        memoryBlocksFile = mydir + "bb.out"

        graph.ReadClusters(clustersAreaFile, 14, 2)
        graph.readClustersInstances(clustersInstancesFile, 0, 0)
        # graph.readMemoryBloc  ks(memoryBlocksFile, 14, 4)
        # Begin with the netWL file, as there are less nets there.
        graph.readNetsWireLength(netsWL, 14, 2)
        graph.readNets(netsInstances, 0, 0)

        graph.findHyperedges()

        edgeWeightType = 0
        graph.computeHyperedgeWeights(True)
        graph.computeVertexWeights()

        paritionFiles = []

        for edgeWeightType in xrange(0, EDGE_WEIGHTS_TYPES):
            for vertexWeightType in xrange(0, VERTEX_WEIGHTS_TYPES):
                metisInput = "input"
                print "================================================="
                if edgeWeightType == 0:
                    print "> Edge weight: # wires"
                    metisInput += "_wires"
                elif edgeWeightType == 1:
                    print "> Edge weight: wire length"
                    metisInput += "_wire-length"
                elif edgeWeightType == 2:
                    print "> Edge weight: 1 / # wires"
                    metisInput += "_1-over-wires"
                elif edgeWeightType == 3:
                    print "> Edge weight: 1 / wire length"
                    metisInput += "_1-over-wire-length"
                elif edgeWeightType == 4:
                    print "> Edge weight: " + str(WEIGHT_COMBILI_COEF) + " * # wires + " + \
                        str(1 - WEIGHT_COMBILI_COEF) + " * wire length"
                    metisInput += "_wires-wire-length"
                elif edgeWeightType == 5:
                    print "> Edge weight: 1 / " + str(WEIGHT_COMBILI_COEF) + " * # wires + " + \
                        str(1 - WEIGHT_COMBILI_COEF) + " * wire length"
                    metisInput += "_1-over-wires-wire-length"
                elif edgeWeightType == 6:
                    print "> Edge weight: 1 (constant)"
                    metisInput += "_1-constant"

                if vertexWeightType == 0:
                    print "> Vertex weight: cluster area"
                    metisInput += "_area"
                elif vertexWeightType == 1:
                    print "> Vertex weight: constant 1"
                    metisInput += "_constant"
                print "================================================="
                metisInput += ".hgr"
                graph.generateMetisInput(metisInput, edgeWeightType, vertexWeightType)
                graph.GraphPartition(metisInput)
                metisPartitionFile = metisInput + ".part.2"
                partitionDirectivesFile = metisInput + ".tcl"
                graph.WritePartitionDirectives(metisPartitionFile, partitionDirectivesFile)
                graph.computePartitionArea(partitionDirectivesFile)
                paritionFiles.append(metisPartitionFile)
        graph.dumpClusters()
        graph.hammingReport(paritionFiles)
        # graph.plotWeights()