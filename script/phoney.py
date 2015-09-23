"""
PHONEY [Pronounce 'poney'], Partitioning of Hypergraph Obviously Not EasY

Want to use it? Then check the 'METIS_PATH' and 'dirs' variables.
"""

"""
TODO
> Object Oriented
> Clean dead code
> Doc
"""

import math
from subprocess import call
import copy

global METIS_PATH
METIS_PATH = "/home/para/dev/metis_unicorn/hmetis-1.5-linux/"
EDGE_WEIGHTS_TYPES = 6
VERTEX_WEIGHTS_TYPES = 1
WEIGHT_COMBILI_COEF = 0.5

def printProgression(current, max):
    if current == max/5:
        print "20%"
    if current == 2*max/5:
        print "40%"
    if current == 3*max/5:
        print"60%"
    if current == 4*max/5:
        print "80%"
    if current == max:
        print "100%"

class Graph():
    def __init__(self):
        self.clusters = [] # list of Cluster objects
        self.ClusterData     = []   # Each element is a list of cluster data
                                    # (Name, Type, InstCount, Boundary1, Boundary2, Area)
                                    # Ordered as in clustersAreaFile.
        self.ConnectData     = []    
        self.ClusterName     = []    
        self.ClusterArea     = []
        self.clusterInstances = []
        self.clusterWeights = []
        self.nets = [] # list of Net objects.
        self.netName           = []
        self.netInstances      = []
        self.netPins = []
        self.netWL = []
        self.hyperedges = []
        self.hyperedgesComprehensive = []
        self.hyperedgeWeights = []      # Each element is a list of weights.
                                        # Each element of those sublists refer to one weight type.
        self.hyperedgeWeightsMax = []   # Maximum weight for each weight type.
                                        # Ordered by weight type.
        self.hyperedgesWL = []  # Each element is a list of the wire length of each net
                                # in the hyperedge at the same index as the list (meaning
                                # hyperedgesWL[0] is for the hyperedge 0.
        self.Vertices        = []
        self.name            = []    
        self.srcCluster      = []    
        self.toClusterDist   = []    
        self.wires           = []    
        self.EdgeConnections = []
        self.EdgeWeights     = []
        self.maxWires        = 0
        self.maxDistance     = 0
        self.maxwXd          = 0
        self.srcNodes        = 0
        self.ClusterCols     = 0
        self.ClusterRows     = 0
        self.ConnectRows     = 0
        self.ConnectCols     = 0
        self.netRows         = 0
        self.col             = 0
        self.LLX         = []
        self.LLY         = []
        self.URX         = []
        self.URY         = []
        self.CX          = []
        self.CY          = []
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
        # print "--------------------------------------------------->"
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
            # self.ClusterData.append(clusterDataRow)
            # self.ClusterName.append(clusterDataRow[0])
            # vvvvv OO
            cluster = Cluster(clusterDataRow[0], i)
            self.clusters.append(cluster)

            lowerBounds = clusterDataRow[3][1:-1].split(",")    # Exclude the first and last
                                                                # chars: '(' and ')'.
            upperBounds = clusterDataRow[4][1:-1].split(",")
            self.clusters[i].setBoundaries(float(lowerBounds[0]), float(lowerBounds[1]),
                float(upperBounds[0]), float(upperBounds[1]))
            
            self.clusters[i].setArea(float(clusterDataRow[5]))

            # ^^^^^^^^ OO

        # for cluster in self.clusters:
        #     print cluster.boudndaries

        # self.ClusterRows = len(self.ClusterName)
        # self.ClusterCols = len(self.ClusterData[0])
        # print (str("\t Clusters: " + str(self.ClusterRows) + "\t Columns read: " + str(self.ClusterCols) ))
        # print "Done!"
        # print "<---------------------------------------------------\n"

    def readClustersInstances(self, filename, hrows, frows):
        # print "--------------------------------------------------->"
        print (str("Reading clusters instances file: " + filename))
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        # First initialize the list of list of instances.
        # Next, we will insert those instances at the index of the right cluster.
        for i in xrange(0, self.ClusterRows):
            self.clusterInstances.append(list())

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

            # Add the instances list at the row index corresponding to the cluster ID.
            # try:
            #     # Get the ID of the corresponding cluster name.
            #     clusterID = self.ClusterName.index(clusterInstancesRow[0])
            # except:
            #     print "Error: the cluster \"" + clusterInstancesRow[0] + "\" in"+filename+"could not be found in the cluster list."
            # else:
            #     del clusterInstancesRow[0]
            #     self.clusterInstances[clusterID] = clusterInstancesRow

            # vvvvvvvvvv OO => findClusterByName(self, clusterName)
            found, clusterID = self.findClusterByName(clusterInstancesRow[0])
            if found:
                del clusterInstancesRow[0]
                for i, instanceName in enumerate(clusterInstancesRow):
                    instance = Instance(instanceName)
                    self.clusters[clusterID].addInstance(instance)
            # ^^^^^^^^^^ OO

        # print "Done!"
        # print "<---------------------------------------------------\n"

    def readNets(self, filename, hrows, frows):
        # print "--------------------------------------------------->"
        print (str("Reading nets file: " + filename))
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        # Remove the header lines
        for i in xrange(0, hrows):
            del lines[0]
        # Remove the footer lines
        for i in xrange(0,frows):
            del lines[-1]

        for i,line in enumerate(lines):
            line = line.strip(' \n')
            line = line.replace('{','')
            line = line.replace('}','')
            netDataRow = line.split()
            # self.netName.append(netDataRow[0])
            # del netDataRow[0] # Remove the net name from the list
            #                     # TODO: could also not delete it and simpy
            #                     # append(netDataRow[1:])
            # self.netInstances.append(netDataRow)

            # vvvvvvvvvvv OO
            found = False
            netID = 0
            while not found and netID < len(self.nets):
                # print self.nets[netID].name
                if self.nets[netID].name == netDataRow[0]:
                    found = True
                else:
                    netID += 1
            if found:
                # print str(netDataRow[0]) + " found"
                del netDataRow[0]
                for j, instanceName in enumerate(netDataRow):
                    instance = Instance(instanceName)
                    self.nets[netID].addInstance(instance)
            # ^^^^^^^^^^^ OO

            printProgression(i, len(lines))

        self.netRows = len(self.netName)
        # print (str("\t Nets: " + str(self.netRows) ))
        # print "Done!"
        # print "<---------------------------------------------------\n"

    def readNetsWireLength(self, filename, hrows, frows):
        # print "--------------------------------------------------->"
        print (str("Reading wire length nets file: " + filename))
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        # First initialize the list of pin count and wire length.
        for i in xrange(0, self.netRows):
            self.netPins.append(0)
            self.netWL.append(0)

        # Remove the header lines
        for i in xrange(0, hrows):
            del lines[0]
        # Remove the footer lines
        for i in xrange(0,frows):
            del lines[-1]

        for i, line in enumerate(lines):
            # line = line.strip(' \n')
            line = " ".join(line.split()) # Remove all extra whitespace characters.
            netDataRow = line.split()

            # try:
            #     netID = self.netName.index(netDataRow[0])
            # except:
            #     print "Error: the net \"" + netDataRow[0] +"\" in " + filename + \
            #         " could not be found in the list of net names."
            # else:
            #     self.netPins[netID] = int(netDataRow[1])
            #     self.netWL[netID] = int(float(netDataRow[2]))

            # vvvvvvvvvv OO
            net = Net(netDataRow[0], i)
            net.setPinAmount(int(netDataRow[1]))
            net.setWL(int(float(netDataRow[2])))
            self.nets.append(net)
            # ^^^^^^^^^^ OO
            printProgression(i, len(lines))

        # print "Done!"
        # print "<---------------------------------------------------\n"
    
    def findHyperedges(self):
        print "Building hyperedges"

        # for netID, nInstances in enumerate(self.netInstances):
        #     connectedClusters = list()

        #     # Now, for each net, we get its list of instances.
        #     for instance in nInstances:

        #         # And for each instance in the net, we check to which cluster
        #         # that particular instance belongs.
        #         for clusterID, cInstances in enumerate(self.clusterInstances):
        #             # Try to find the instance from the net in the cluster
        #             try:
        #                 cInstanceIndex = cInstances.index(instance)
        #             except:
        #                 continue
        #             else:
        #                 # If found, see if the cluster has already been added
        #                 # to connectedClusters.
        #                 try:
        #                     connectedClusterIndex = connectedClusters.index(clusterID)
        #                 except:
        #                     connectedClusters.append(clusterID)

        #     if len(connectedClusters) > 1:
        #         listID = [netID] # We want the netID to be inserted in the hyperedge row as a list.
        #         connectedClusters.insert(0, listID)
        #         self.hyperedges.append(connectedClusters)

        # vvvvvvvvvvvvvv OO
        # print "len(self.clusters) = " + str(len(self.clusters))
        # print "len(self.nets) = " + str(len(self.nets))
        for i, net in enumerate(self.nets):
            connectedClusters = list()
            # print "len(net.instances) = " + str(len(net.instances))

            # Now, for each net, we get its list of instances.
            for netInstance in net.instances: # netInstance are Instance object.

                # And for each instance in the net, we check to which cluster
                # that particular instance belongs.
                for cluster in self.clusters:

                    # Try to find the instance from the net in the cluster
                    if cluster.searchInstance(netInstance):
                        # If found, see if the cluster has already been added
                        # to connectedClusters.
                        j = 0
                        clusterFound = False
                        while j < len(connectedClusters) and not clusterFound:
                            # print type(connectedClusters[j])
                            if connectedClusters[j].ID == cluster.ID:
                                clusterFound = True
                            else:
                                j += 1
                        if not clusterFound:
                            connectedClusters.append(cluster)
                        # try:
                        #     connectedClusterIndex = connectedClusters.index(cluster.ID)
                        # except:
                        #     connectedClusters.append(cluster)

            # Append the list A of connected clusters to the list B of hyperedges
            # only if there are more than one cluster in list A.
            if len(connectedClusters) > 1:
                # print "len(connectedClusters) > 1"
                hyperedge = Hyperedge()
                for cluster in connectedClusters:
                    hyperedge.addCluster(cluster)
                hyperedge.addNet(net)
                self.hyperedges.append(hyperedge)

            printProgression(i, len(self.nets))
        # print "len(self.hyperedges) = " + str(len(self.hyperedges))
        # ^^^^^^^^^^^^^^ OO


        for hyperedge in self.hyperedges:
            print "Hyperedge:"
            for cluster in hyperedge.clusters:
                print cluster.ID

        # self.hyperedgesComprehensive = copy.deepcopy(self.hyperedges) # Keep the comprehensive data somewhere, just in case.

        # print "Before merger:"
        # print self.hyperedges
        # Merge duplicates
        print "Merging hyperedges"
        # for i, hyperedge in enumerate(self.hyperedges):
        #     duplicate = False
        #     j = i
        #     while j < len(self.hyperedges):
        #         if len(self.hyperedges[j]) == len(hyperedge) and j != i:
        #             duplicate = True
        #             # Check if all clusters in the hyperedge are the same
        #             for k in xrange(1,len(self.hyperedges[j])):
        #                 if self.hyperedges[j][k] != hyperedge[k]:
        #                     duplicate = False

        #             if duplicate:
        #                 # Append the duplicate hyperedge net ID to the list.
        #                 # Beware it's already (alone) in a list, so we need to access
        #                 # the first element, hence that
        #                 # [j=hyperedge][0=list of IDs][0=net ID]
        #                 self.hyperedges[i][0].append(self.hyperedges[j][0][0])
        #                 del self.hyperedges[j]
        #             else:
        #                 j += 1
        #         else:
        #             j += 1

        # vvvvvvvvvvvvvvv OO
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
                        del self.hyperedges[j]
                    else:
                        j += 1


                # If hyperedgeA is smaller than hyperedgeB, check if
                # A can be included in B.
                elif len(hyperedgeA.clusters) < len(hyperedgeB.clusters):
                    discard = False
                    k = 0 # index for A's clusters
                    while k < len(hyperedgeA.clusters) and not discard:
                        included = False    # Reset for each loop, we want to know if
                                            # every clusters from A are included in B.
                        l = 0 # index for B's clusters
                        while l < len(hyperedgeB.clusters) and not included:
                            # print type(hyperedgeA.clusters[k])
                            # print type(hyperedgeB.clusters[l])
                            if hyperedgeA.clusters[k].name == hyperedgeB.clusters[l].name:
                                included = True
                            l += 1
                        if not included:
                            discard = True
                        else:
                            k += 1
                    # The inclusion was not successful
                    if discard:
                        j += 1
                    # Inclusion possible. Do it. Naow.
                    else:
                        clusterAMerged = True
                        for net in hyperedgeA.nets:
                            hyperedgeB.addNet(net)
                        del self.hyperedges[i] # HyperedgeA


                # If hyperedgeA is larger than hyperedgeB, check if
                # B can be included in A.
                elif len(hyperedgeA.clusters) > len(hyperedgeB.clusters):
                    discard = False
                    l = 0
                    while l < len(hyperedgeB.clusters) and not discard:
                        included = False
                        k = 0
                        while k < len(hyperedgeA.clusters) and not included:
                            if hyperedgeB.clusters[l].name == hyperedgeA.clusters[k].name:
                                included = True
                            k += 1
                        if not included:
                            discard = True
                        else:
                            l += 1
                    if discard:
                        j += 1
                    else:
                        for net in hyperedgeB.nets:
                            hyperedgeA.addNet(net)
                        del self.hyperedges[j] # HyperedgeB

                else:
                    j += 1

            # If hyperedgeA has not been merged, inspect the next one.
            # Otherwise, hyperedgeA has been deleted and all the following
            # elements have been shifted, thus no need to increment the index.
            if not clusterAMerged:
                i += 1

            printProgression(i, len(self.hyperedges))
        # ^^^^^^^^^^^^^^^ OO

        # print "After duplicates:"
        # print self.hyperedges

        # Next: merge the inclusions.
        # print "Merging hyperedge inclusions"
        # i = 0
        # while i < len(self.hyperedges):
        #     hyperedge = self.hyperedges[i]
        #     j = 0
        #     included = False
        #     while j < len(self.hyperedges) and not included:
        #         if j == i:
        #             j += 1
        #         else:
        #             discard = False
        #             k = 1
        #             while not discard and k < len(hyperedge):
        #                 try:
        #                     clusterID = self.hyperedges[j][1:].index(hyperedge[k])
        #                 except:
        #                     discard = True
        #                     j += 1
        #                 else:
        #                     k += 1

        #                 if k == len(hyperedge):
        #                     included = True

        #             if included:
        #                 for net in hyperedge[0]:
        #                     self.hyperedges[j][0].append(net)
        #                 del self.hyperedges[i]

        #     if not included:
        #         i += 1

        # print "After merger:"
        # print self.hyperedges





    def generateMetisInput(self, filename, edgeWeightType):
        print "Generating METIS input file..."
        s = str(len(self.hyperedges)) + " " + str(len(self.clusters)) + " 11"
        for i, hyperedge in enumerate(self.hyperedges):
            s += "\n" + str(hyperedge.weightsNormalized[edgeWeightType]) + " "
            for cluster in hyperedge.clusters:
                s += str(cluster.ID) + " "
        for cluster in self.clusters:
            s += "\n" + str(cluster.weights[0])
        # print s
        with open(filename, 'w') as file:
            file.write(s)


    def computeHyperedgeWeights(self, normalized):
        print "Generating weights of hyperedges."

        self.hyperedgeWeightsMax = [0] * EDGE_WEIGHTS_TYPES
        for i, hyperedge in enumerate(self.hyperedges):
            # self.hyperedgeWeights.append(list())
            # self.hyperedgeWeights[i] = [0] * EDGE_WEIGHTS_TYPES
            for weightType in xrange(0, EDGE_WEIGHTS_TYPES):
                weight = 0
                if weightType == 0:
                    # Number of wires
                    for net in hyperedge.nets:
                        weight += net.pins
                elif weightType == 1:
                    # Wire length
                    for net in hyperedge.nets:
                        weight += net.wl
                elif weightType == 2:
                    # 1/#wires
                    weight = 1.0 / hyperedge.weights[0]
                elif weightType == 3:
                    # 1/Wire length
                    weight = 1.0 / hyperedge.weights[1]
                elif weightType == 4:
                    # wire length and number of wires
                    weight = \
                        WEIGHT_COMBILI_COEF * hyperedge.weights[0] + \
                        (1 - WEIGHT_COMBILI_COEF) * hyperedge.weights[1]
                elif weightType == 5:
                    # 1 / (wire length and number of wires)
                    weight = 1.0 / ( \
                        WEIGHT_COMBILI_COEF * hyperedge.weights[0] + \
                        (1 - WEIGHT_COMBILI_COEF) * hyperedge.weights[1] )

                hyperedge.setWeight(weightType, weight) # OO
                # self.hyperedgeWeights[i][weightType] = weight
                # Save the max
                if weight > self.hyperedgeWeightsMax[weightType]:
                    self.hyperedgeWeightsMax[weightType] = weight

        print self.hyperedgeWeightsMax

        # Normalization
        if normalized:
            # # Find max
            # maxWeight = 0
            # for weight in self.hyperedgeWeightsMax:
            #     if weight > maxWeight:
            #         maxWeight = weight
            maxWeight = 1000
            # Normalize
            # for i in xrange(0, len(self.hyperedgeWeights)):
            #     for j in xrange(0, EDGE_WEIGHTS_TYPES):
            #         # There is no need to normalize the weights from which we chose the maximum weight (obviously)
            #         if maxWeight != self.hyperedgeWeightsMax[j]:
            #             self.hyperedgeWeights[i][j] = int((self.hyperedgeWeights[i][j] * maxWeight) / self.hyperedgeWeightsMax[j])
            # vvvvvvvvvvv OO
            for hyperedge in self.hyperedges:
                for i in xrange(0, EDGE_WEIGHTS_TYPES):
                    hyperedge.setWeightNormalized(i, int((hyperedge.weights[i] * maxWeight) / self.hyperedgeWeightsMax[i]))
            # ^^^^^^^^^^^ OO

        # print self.hyperedgeWeights



    def computeVertexWeights(self):
        print "Generating weights of vertex."
        for i, cluster in enumerate(self.ClusterData):
            self.clusterWeights.append(cluster[5])

        # vvvvvvvvvv OO
        for hyperedge in self.hyperedges:
            for cluster in hyperedge.clusters:
                cluster.setWeight(0, cluster.area)
        # ^^^^^^^^^^ OO



    # def computeHyperedgeStat(self):
    #     print "Generating statistics on the hyperedges."
    #     for i, hyperedge in enumerate(self.hyperedges)
    #         self.hyperedgesWL.append(list())
    #         for net in hyperedge:
    #             self.hyperedgesWL[i].append(self.netWL[net])
    #     print "Hyperedges statistics:"
    #     print "Hyperedge \t WL tot \t WL max \t WL avg \t WL std dev"
    #     for i, hyperedge in enumerate(self.hyperedges):
    #         print str(i)
    #         print 














    def ReadConnectivity(self, filename, hrows, frows):
        print "--------------------------------------------------->"
        print (str("Reading connectivity file: " + filename))
        f = open(filename)
        lines = f.readlines()
        alllines=len(lines)
        f.close()
        count=1
        for line in lines:
            if (count > hrows) and (count < (alllines - frows + 1)):
                t=line.rstrip()
                tt = t.split()
                self.ConnectData.append(tt)
                self.name.append(tt[0])
                self.ConnectRows +=1
            count +=1
        self.ConnectCols=len(self.ConnectData[0]) 
        print (str("\t Edges: " + str(self.ConnectRows) + "\t Columns read: " + str(self.ConnectCols)))
        print "Done!"
        print "<---------------------------------------------------\n"
    
    def SplitComma(self):
        for line in self.ClusterData:
            LL = line[3].strip("()")
            UR = line[4].strip("()")
            LLsplit = LL.split(",")
            URsplit = UR.split(",")
            self.LLX.append(LLsplit[0])
            self.LLY.append(LLsplit[1])
            self.URX.append(URsplit[0])
            self.URY.append(URsplit[1])
            tmpCX=abs(float(URsplit[0])-(float(URsplit[0]) - float(LLsplit[0]))/2)
            tmpCY=abs(float(URsplit[1])-(float(URsplit[1]) - float(LLsplit[1]))/2)
            self.CX.append(str(tmpCX))
            self.CY.append(str(tmpCY))
            self.ClusterArea.append(str(line[5]))

    def BuildEdges(self):
        print "--------------------------------------------------->"
        # Check for Black Box that is taken as 2 columns 
        # Check for line ends
        print (str("Building connectivity..."))
        print "\t N of nodes: " + str(self.ClusterRows)
        print "\t N of edges: " + str(self.ConnectRows)
        self.WriteLog("From building edges: ")
        ### This will work only on simple edges
        ### Scan according to edges 
        count       = 0
        maxWires    = 0
        maxDistance = 0
        maxwXd      = 0
        totalEdgeLength = 0
        totalEdgeWL = 0
        for connection in self.ConnectData:
            tmp = []
            SRC = connection[0]
            DST = connection[3]
            E12E2 = int(connection[6])
            E22E1 = int(connection[7])
            E1E2  = int(connection[8])
            wires = E12E2 + E22E1 + E1E2 
            II = self.ClusterName.index(SRC)
            KK = self.ClusterName.index(DST)
            distance = float ( math.sqrt (\
            (float(self.CX[II]) - float(self.CX[KK])) * (float(self.CX[II]) - float(self.CX[KK])) + \
            (float(self.CY[II]) - float(self.CY[KK])) * (float(self.CY[II]) - float(self.CY[KK])))  )
            wXd = float(distance) * float(wires)
            tmp.append(SRC)
            tmp.append(DST)
            tmp.append(wires)
            tmp.append(distance)
            totalEdgeLength += float(distance)
            totalEdgeWL += float(wXd)
            tmp.append(wXd)
            if (distance > maxDistance):
                maxDistance = distance 
            if (wires > maxWires):
                maxWires= wires  
            if (wXd > maxwXd):
                maxwXd = wXd  
            self.EdgeConnections.append(tmp)
            msg = "SRC :" + str(SRC) + "\t DST: " + str(DST) + "\t Wires.: " + str(wires) + "\t Dist.: " + str(distance)
            self.WriteLog(msg)
        self.maxWires    = maxWires
        self.maxDistance = maxDistance 
        self.maxwXd      = maxwXd 
        #self.WriteLog(self.EdgeConnections)
        msg = "Max wires :" + str(self.maxWires) + "\t Max dist: " + str(self.maxDistance) + "\t Max wXd: " + str(self.maxwXd)
        self.WriteLog(msg)
        msg = "Total edge length:" + str(totalEdgeLength) + "\t Total WL: " + str(totalEdgeWL)  
        self.WriteLog(msg)
        for connection in self.EdgeConnections:
            self.WriteLog(str("SRC: " + connection[0] + "\t DST: " + connection[1]))
            found1=0
            found2=0
            for vertice in self.Vertices: 
                if (vertice == connection[0]):
                    found1=1
                if (vertice == connection[1]):
                    found2=1
            if (found1 == 0) :
                self.Vertices.append(connection[0])
            if (found2 == 0):
                self.Vertices.append(connection[1])
        i=0
        for vertice in self.Vertices:
            self.WriteLog(str('%5s' % str(i) +"\t"+ str(vertice)))
            i+=1
        print "Actual vertices used: ", i-1
        print "Done!"
        print "<---------------------------------------------------\n"

    def BuildEdgeWeight(self):
        print "--------------------------------------------------->"
        print "Building edge weights:"
        tmp = []
        for edge in self.EdgeConnections:        
            # 0. N wires
            tmp.append("%.0f" % float(float(edge[2]) / float(self.maxWires) * 1000 + 1))
            # 1. distance
            tmp.append("%.0f" % float(float(edge[3]) / float(self.maxDistance) * 10000 + 1 ))
            # 2. wXd
            tmp.append("%.0f" % float(float(edge[4]) / float(self.maxwXd) * 1000 + 1 ))
            # 3. 1/ N wires
        #    tmp.append("%.3f" % float(1/float(edge[2])*float(self.maxWires)))
            tmp.append("%.0f" % float(1/float(edge[2]) * 1000 + 1))
            # 4. 1/ distance
            #tmp.append("%.3f" % float(1/float(edge[3])*float(self.maxDistance)))
            tmp.append("%.0f" % float(1/(float(edge[3]) + 0.01) * 1000 + 1))
            # 5. 1/ wXd
            #tmp.append("%.3f" % float(1/float(edge[4])*float(self.maxwXd)))
            tmp.append("%.0f" % float(1/(float(edge[4]) + 0.01) * 1000 + 1))
            self.EdgeWeights.append(tmp)        
            tmp = []
        #print self.EdgeWeights 
        print "<---------------------------------------------------"
        print "Done!"

#+++>
    def WriteMetisFile(self, CostFunction):
        print "--------------------------------------------------->"
        print "Write output file for cost:" , CostFunction 
        # Extract graph in .graph format for METIS and write file
        filename ="partition" + str(CostFunction) + ".hgr"
        f1 = open(filename,'w')
        # Write preambule
        f1.write(str(self.ConnectRows) + " "  + str(len(self.Vertices)) + " " + "11" + "\n")
        cost = CostFunction
        for edge in self.EdgeConnections:        
            ii = self.EdgeConnections.index(edge)
            SRC = self.EdgeConnections[ii][0]
            DST = self.EdgeConnections[ii][1]
            tmp ="Weight: " + str(self.EdgeWeights[ii][0]) + '%9s' % "\t SRC: " + str(self.Vertices.index(SRC)) + '%9s' % "\t DST: " + str(self.Vertices.index(DST)) 
            self.WriteLog(tmp)
            # write the appropriate cost for edges
            f1.write(str(self.EdgeWeights[ii][cost]) + " " + str(int(self.Vertices.index(SRC)) + 1) + " " + str(int(self.Vertices.index(DST)) + 1) + "\n")
        for vertice in self.Vertices:
           II = self.ClusterName.index(vertice) 
           f1.write(str(self.ClusterArea[II]) + "\n")
        f1.close()            
        print "<---------------------------------------------------"
        print "Done!"
#+++>
    def PrintData(self,mode):
        f = open('output.rpt', 'w')
        if(mode == 'v'): 
            print "Clusters: " ,  self.ClusterRows
            print self.toCluster
        #for j in range (0, self.ClusterRows):
        for j in range (0, len(self.toCluster)):
            if (self.toCluster[j] != []):
                txt = (str("Cluster: "+self.ClusterName[j]))
                f.write(txt);
                if(mode == 'v'): print(txt)
                txt="\nis connected to clusters: \t" 
                f.write(txt);
                if (mode == 'v'): print(txt)
                txt = str(self.toCluster[j])  
                f.write(txt)
                if (mode == 'v'): print(txt)
                txt = "\nwith no of wires: \t" 
                f.write(txt)
                if (mode == 'v'): print(txt)
                txt = str(self.Wires[j]) 
                f.write(txt)
                if (mode == 'v'): print(txt)
                txt = "\nwith distances : \t" 
                f.write(txt)
                if (mode == 'v'): print(txt)
                txt = str(self.toClusterDist[j]) 
                f.write(txt)
                if (mode == 'v'): print(txt)
                txt = "\n" 
                f.write(txt)
                if (mode == 'v'): print(txt)
                txt = "At ("+str(self.CX[j])+","+ str(self.CY[j])+")\t"  
                f.write(txt)
                if (mode == 'v'): print(txt)
                txt = "(LLX,LLY) ("+ str(self.LLX[j])+","+str(self.LLY[j])+")\t"   
                f.write(txt)
                if (mode == 'v'): print(txt)
                txt = "(URX,URY) ("+str(self.URX[j])+","+str(self.URY[j])+")\n"   
                f.write(txt)
                if (mode == 'v'): print(txt)
                txt = "\n\n" 
        f.close()

    def GraphPartition(self, filename):
        print "--------------------------------------------------->"
        # print "Running partition on cost: ", CostFunction
        print "Running hmetis with " + filename
        # call(["/Users/drago/bin/hmetis-1.5-osx-i686/hmetis",filename,"2","5","20","1","1","1","0","0"])
        command = METIS_PATH + "hmetis " + filename + " 2 5 20 1 1 1 0 0"
        print "Calling '" + command + "'"
        # call([METIS_PATH + "hmetis",filename,"2","5","20","1","1","1","0","0"])
        call(command.split())
    
    def WritePartitionDirectives(self, metisFile):
        # print "--------------------------------------------------->"
        print "Write tcl file for file: ", metisFile
        filenameOut = metisFile + ".tcl"
        filenameIn = metisFile + ".part.2"
        try:
            fOut = open(filenameOut, "w")
            fIn = open(filenameIn, "r")
        except IOError as e:
            print "Can't open file"
            return False
        data = fIn.readlines()
        for i, cluster in enumerate(self.ClusterData):
            fOut.write(str('add_to_die -cluster ' + "\t" + str(cluster[0]) + \
                "\t -die Die" + str(data[i])))
        fOut.close()
        fIn.close()
        print "Done!"
        # print "<---------------------------------------------------\n"


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


class Cluster:
    def __init__(self, name, clusterID):
        self.name = name
        self.ID = clusterID
        self.instances = []
        self.boudndaries = [[0, 0], [0 ,0]] # [[lower X, lower U], [upper X, upper Y]] (floats)
        self.area = 0 # float
        self.weights = []   # [0] = area
                            # [1] = power
                            # [2] = area & power
        self.weightsNormalized = []

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
        i = 0
        while not found and i < len(self.instances):
            if self.instances[i].name == instance.name:
                found = True
            else:
                i += 1
        return found

    def setArea(self, area):
        self.area = area

    def setWeight(self, index, weight):
        self.weights[index] = weight

    def setWeightNormalized(self, index, weight):
        self.weightsNormalized[index] = weight


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
#    dirs=["/Users/drago/Desktop/Current/test/inpt/CCX/ClusterLevel2/"]
    # dirs=["/Users/drago/Desktop/Current/test/inpt/CCX/ClusterLevel3/"]
    dirs=["../input_files/"]
    # dirs=["../ccx/"]
#    dirs=["/Users/drago/Desktop/Current/test/inpt/SPC/"]
#    dirs=["/Users/drago/Desktop/Current/test/inpt/test/"]

    graph = Graph()
#    RunCosts = [0, 1, 2]
#    RunCosts = [3, 4, 5]
    RunCosts = [0, 1, 2, 3, 4, 5]

    for mydir in dirs:
        clustersAreaFile = mydir + "ClustersArea.out"
        clustersInstancesFile = mydir + "ClustersInstances.out"
        netsInstances = mydir + "InstancesPerNet.out"
        netsWL = mydir + "WLnets.out"

        graph.ReadClusters(clustersAreaFile, 14, 2)
        graph.readClustersInstances(clustersInstancesFile, 0, 0)
        # Begin with the netWL file, as there are less nets there.
        graph.readNetsWireLength(netsWL, 14, 2)
        graph.readNets(netsInstances, 0, 0)

        graph.findHyperedges()

        edgeWeightType = 0
        graph.computeHyperedgeWeights(True)
        graph.computeVertexWeights()

        for edgeWeightType in xrange(0, EDGE_WEIGHTS_TYPES):
            metisInput = "input"
            for vertexWeightType in xrange(0, VERTEX_WEIGHTS_TYPES):
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

                if vertexWeightType == 0:
                    print "> Vertex weight: cluster area"
                    metisInput += "_area"
                print "================================================="
                metisInput += ".hgr"
                graph.generateMetisInput(metisInput, edgeWeightType)
                graph.GraphPartition(metisInput)
                graph.WritePartitionDirectives(metisInput)


