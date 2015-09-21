"""
PHONEY [Pronounce 'poney'], Partitioning of Hypergraph Obviously Not EasY
"""


import math
from subprocess import call
import copy

global METIS_PATH
METIS_PATH = "/home/para/dev/metis_unicorn/hmetis-1.5-linux/"
EDGE_WEIGHTS_TYPES = 5
VERTEX_WEIGHTS_TYPES = 1
WEIGHT_COMBILI_COEF = 0.5

class Graph():
    def __init__(self):
        self.ClusterData     = []   # Each element is a list of cluster data
                                    # (Name, Type, InstCount, Boundary1, Boundary2, Area)
                                    # Ordered as in clustersAreaFile.
        self.ConnectData     = []    
        self.ClusterName     = []    
        self.ClusterArea     = []
        self.clusterInstances = []
        self.clusterWeights = []
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

        for line in lines:
            line = line.strip(' \n')
            clusterDataRow = line.split()
            self.ClusterData.append(clusterDataRow)
            self.ClusterName.append(clusterDataRow[0])

        self.ClusterRows = len(self.ClusterName)
        self.ClusterCols = len(self.ClusterData[0])
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
            try:
                # Get the ID of the corresponding cluster name.
                clusterID = self.ClusterName.index(clusterInstancesRow[0])
            except:
                print "Error: the cluster \"" + clusterInstancesRow[0] + "\" in"+filename+"could not be found in the cluster list."
            else:
                del clusterInstancesRow[0]
                self.clusterInstances[clusterID] = clusterInstancesRow

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

        for line in lines:
            line = line.strip(' \n')
            line = line.replace('{','')
            line = line.replace('}','')
            netDataRow = line.split()
            self.netName.append(netDataRow[0])
            del netDataRow[0] # Remove the net name from the list
            self.netInstances.append(netDataRow)

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

        for line in lines:
            # line = line.strip(' \n')
            line = " ".join(line.split()) # Remove all extra whitespace characters.
            netDataRow = line.split()

            try:
                netID = self.netName.index(netDataRow[0])
            except:
                print "Error: the net \"" + netDataRow[0] +"\" in " + filename + \
                    " could not be found in the list of net names."
            else:
                self.netPins[netID] = int(netDataRow[1])
                self.netWL[netID] = int(float(netDataRow[2]))

        # print "Done!"
        # print "<---------------------------------------------------\n"
    
    def findHyperedges(self):
        print "Building hyperedges"

        for netID, nInstances in enumerate(self.netInstances):
            connectedClusters = list()

            # Now, for each net, we get its list of instances.
            for instance in nInstances:

                # And for each instance in the net, we check to which cluster
                # that particular instance belongs.
                for clusterID, cInstances in enumerate(self.clusterInstances):
                    # Try to find the instance from the net in the cluster
                    try:
                        cInstanceIndex = cInstances.index(instance)
                    except:
                        continue
                    else:
                        # If found, see if the cluster has already been added
                        # to connectedClusters.
                        try:
                            connectedClusterIndex = connectedClusters.index(clusterID)
                        except:
                            connectedClusters.append(clusterID)

            if len(connectedClusters) > 1:
                listID = [netID] # We want the netID to be inserted in the hyperedge row as a list.
                connectedClusters.insert(0, listID)
                self.hyperedges.append(connectedClusters)

        # for hyperEdge in self.hyperedges:
        #     print hyperEdge
        self.hyperedgesComprehensive = copy.deepcopy(self.hyperedges) # Keep the comprehensive data somewhere, just in case.


        # print "Before merger:"
        # print self.hyperedges
        # Merge duplicates
        for i, hyperedge in enumerate(self.hyperedges):
            duplicate = False
            j = i
            while j < len(self.hyperedges):
                if len(self.hyperedges[j]) == len(hyperedge) and j != i:
                    duplicate = True
                    # Check if all clusters in the hyperedge are the same
                    for k in xrange(1,len(self.hyperedges[j])):
                        if self.hyperedges[j][k] != hyperedge[k]:
                            duplicate = False

                    if duplicate:
                        # Append the duplicate hyperedge ID to the list.
                        # Beware it's already (alone) in a list, so we need to access
                        # the first element, hence that
                        # [j=hyperedge][0=list of IDs][0=ID]
                        self.hyperedges[i][0].append(self.hyperedges[j][0][0])
                        del self.hyperedges[j]
                    else:
                        j += 1
                else:
                    j += 1

        # print "After duplicates:"
        # print self.hyperedges

        # Next: merge the inclusions.
        i = 0
        while i < len(self.hyperedges):
            hyperedge = self.hyperedges[i]
            j = 0
            included = False
            while j < len(self.hyperedges) and not included:
                if j == i:
                    j += 1
                else:
                    discard = False
                    k = 1
                    while not discard and k < len(hyperedge):
                        try:
                            clusterID = self.hyperedges[j][1:].index(hyperedge[k])
                        except:
                            discard = True
                            j += 1
                        else:
                            k += 1

                        if k == len(hyperedge):
                            included = True

                    if included:
                        for net in hyperedge[0]:
                            self.hyperedges[j][0].append(net)
                        del self.hyperedges[i]

            if not included:
                i += 1

        # print "After merger:"
        # print self.hyperedges





    def generateMetisInput(self, filename, edgeWeightType):
        print "Generating METIS input file..."
        s = str(len(self.hyperedges)) + " " + str(len(self.ClusterName)) + " 11"
        for i, hyperedge in enumerate(self.hyperedges):
            s += "\n" + str(self.hyperedgeWeights[i][edgeWeightType]) + " "
            for cluster in hyperedge[1:]:
                s += str(cluster) + " "
        for weight in self.clusterWeights:
            s += "\n" + str(weight)
        # print s
        with open(filename, 'w') as file:
            file.write(s)


    def computeHyperedgeWeights(self, normalized):
        print "Generating weights of hyperedges."

        self.hyperedgeWeightsMax = [0] * EDGE_WEIGHTS_TYPES
        for i, hyperedge in enumerate(self.hyperedges):
            self.hyperedgeWeights.append(list())
            self.hyperedgeWeights[i] = [0] * EDGE_WEIGHTS_TYPES
            for weightType in xrange(0, EDGE_WEIGHTS_TYPES):
                if weightType == 0:
                    # Number of wires
                    for net in hyperedge[0]:
                        self.hyperedgeWeights[i][weightType] += self.netPins[net]
                elif weightType == 1:
                    # Wire length
                    for net in hyperedge[0]:
                        self.hyperedgeWeights[i][weightType] += self.netWL[net]
                elif weightType == 2:
                    # 1/#wires
                    self.hyperedgeWeights[i][weightType] = 1.0 / self.hyperedgeWeights[i][0]
                elif weightType == 3:
                    # 1/Wire length
                    self.hyperedgeWeights[i][weightType] = 1.0 / self.hyperedgeWeights[i][1]
                elif weightType == 4:
                    # wire length and number of wires
                    self.hyperedgeWeights[i][weightType] = \
                        WEIGHT_COMBILI_COEF * self.hyperedgeWeights[i][0] + \
                        (1 - WEIGHT_COMBILI_COEF) * self.hyperedgeWeights[i][1]
                elif weightType == 5:
                    # 1 / (wire length and number of wires)
                    self.hyperedgeWeights[i][weightType] = 1.0 / ( \
                        WEIGHT_COMBILI_COEF * self.hyperedgeWeights[i][0] + \
                        (1 - WEIGHT_COMBILI_COEF) * self.hyperedgeWeights[i][1]
                # Save the max
                if self.hyperedgeWeights[i][weightType] > self.hyperedgeWeightsMax[weightType]:
                    self.hyperedgeWeightsMax[weightType] = self.hyperedgeWeights[i][weightType]

        print self.hyperedgeWeights

        # Normalization
        if normalized:
            # Find max
            maxWeight = 0
            for weight in self.hyperedgeWeightsMax:
                if weight > maxWeight:
                    maxWeight = weight
            # Normalize
            for i in xrange(0, len(self.hyperedgeWeights)):
                for j in xrange(0, EDGE_WEIGHTS_TYPES):
                    # There is no need to normalize the weights from which we chose the maximum weight (obviously)
                    if maxWeight != self.hyperedgeWeightsMax[j]:
                        self.hyperedgeWeights[i][j] = int((self.hyperedgeWeights[i][j] * maxWeight) / self.hyperedgeWeightsMax[j])
        print self.hyperedgeWeights



    def computeVertexWeights(self):
        print "Generating weights of vertex."
        for i, cluster in enumerate(self.ClusterData):
            self.clusterWeights.append(cluster[5])














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
    
    def WritePartitionDirectives(self, CostFunction):
        print "--------------------------------------------------->"
        print "Write tcl file for cost: ", CostFunction
        filename ="Partition" + str(CostFunction) + ".tcl"
        filename2 ="partition" + str(CostFunction) + ".hgr.part.2"
        try:
            f = open(filename, "w")
            f2 = open(filename2, "r")
        except IOError as e:
            print "Can't open file"
            return False
        data = f2.readlines()
        i=0
        for vertice in self.Vertices:
            f.write(str('add_to_die -cluster ' + "\t"+ str(vertice) + "\t -die Die" + str(data[i])))
            i+=1
        f.close()
        f2.close()
        print "Done!"
        print "<---------------------------------------------------\n"

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
        metisInput = "input.hgr"
        # connect_file = mydir + "connectivity.rpt"
        graph.ReadClusters(clustersAreaFile, 14, 2)
        graph.readClustersInstances(clustersInstancesFile, 0, 0)
        graph.readNets(netsInstances, 0, 0)
        graph.readNetsWireLength(netsWL, 14, 2)

        graph.findHyperedges()

        edgeWeightType = 0
        graph.computeHyperedgeWeights(True)
        graph.computeVertexWeights()

        for edgeWeightType in xrange(0, EDGE_WEIGHTS_TYPES):
            for vertexWeightType in xrange(0, VERTEX_WEIGHTS_TYPES):
                print "================================================="
                if edgeWeightType == 0:
                    print "> Edge weight: # wires"
                elif edgeWeightType == 1:
                    print "> Edge weight: wire length"
                elif edgeWeightType == 2:
                    print "> Edge weight: 1 / # wires"
                elif edgeWeightType == 3:
                    print "> Edge weight: 1 / wire length"
                elif edgeWeightType == 4:
                    print "> Edge weight: " + str(WEIGHT_COMBILI_COEF) + " * # wires + " + \
                        str(1 - WEIGHT_COMBILI_COEF) + " * wire length"

                if vertexWeightType == 0:
                    print "> Vertex weight: cluster area"
                print "================================================="
                graph.generateMetisInput(metisInput, edgeWeightType)
                graph.GraphPartition(metisInput)

        # for i, cluster in enumerate(graph.ClusterName):
        #     str = cluster
        #     str += "\t" + graph.ClusterData[i][4]
        #     print str



        # graph.setVertexWeight()

        # graph.ReadConnectivity(connect_file, 34, 2)
        # graph.SplitComma()
        # graph.BuildEdges() 
        # graph.BuildEdgeWeight() 
        # for run in RunCosts:
        #     graph.WriteMetisFile(run)
        # for run in RunCosts:
        #     graph.GraphPartition(run)
        # for run in RunCosts:
        #     graph.WritePartitionDirectives(run)
