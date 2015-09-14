import math
from subprocess import call

class Graph():
    def __init__(self):
        self.ClusterData     = []    
        self.ConnectData     = []    
        self.ClusterName     = []    
        self.ClusterArea     = []    
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
        print "--------------------------------------------------->"
        print (str("Reading clusters file : " + filename))
        f = open(filename)
        lines = f.readlines()
        alllines=len(lines)
        f.close()
        count=1
        for line in lines:
            if (count > hrows) and (count < (alllines - frows + 1)):
                t=line.rstrip()
                tt = t.split()
                self.ClusterData.append(tt)
                self.ClusterName.append(tt[0])
                self.ClusterRows +=1
            count +=1
        self.ClusterCols=len(self.ClusterData[0]) 
        print (str("\t Clusters : " + str(self.ClusterRows) + "\t Columns read : " + str(self.ClusterCols) ))
        print "Done!"
        print "<---------------------------------------------------\n"
    
    def ReadConnectivity(self, filename, hrows, frows):
        print "--------------------------------------------------->"
        print (str("Reading connectivity file : " + filename))
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
        print (str("\t Edges : " + str(self.ConnectRows) + "\t Columns read : " + str(self.ConnectCols)))
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

    def GraphPartition(self, CostFunction):
        print "--------------------------------------------------->"
        print "Running partition on cost: ", CostFunction
         #~/bin/hmetis-1.5-osx-i686/hmetis partition3.hgr 2 5 20 1 1 1 0 0
        filename ="partition" + str(CostFunction) + ".hgr"
        call(["/Users/drago/bin/hmetis-1.5-osx-i686/hmetis",filename,"2","5","20","1","1","1","0","0"])
    
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
#
#    def GenGraph(self,mode):
#        f = open('output.rpt', 'w')
#        if(mode == 'v'): 
#            print "Clusters: " ,  self.rows
#        for j in range (0, self.rows):
#            if (self.toCluster[j] != []):
#                txt = (str("Cluster: "+self.name[j]))
#                f.write(txt);
#                if(mode == 'v'): print(txt)
#                txt="\nis connected to clusters: \t" 
#                f.write(txt);
#                if (mode == 'v'): print(txt)
#                txt = str(self.toCluster[j])  
#                f.write(txt)
#                if (mode == 'v'): print(txt)
#                txt = "\nwith no of wires: \t" 
#                f.write(txt)
#                if (mode == 'v'): print(txt)
#                txt = str(self.Wires[j]) 
#                f.write(txt)
#                if (mode == 'v'): print(txt)
#                txt = "\nwith distances : \t" 
#                f.write(txt)
#                if (mode == 'v'): print(txt)
#                txt = str(self.toClusterDist[j]) 
#                f.write(txt)
#                if (mode == 'v'): print(txt)
#                txt = "\n" 
#                f.write(txt)
#                if (mode == 'v'): print(txt)
#                txt = "At ("+str(self.CX[j])+","+ str(self.CY[j])+")\t"  
#                f.write(txt)
#                if (mode == 'v'): print(txt)
#                txt = "(LLX,LLY) ("+ str(self.LLX[j])+","+str(self.LLY[j])+")\t"   
#                f.write(txt)
#                if (mode == 'v'): print(txt)
#                txt = "(URX,URY) ("+str(self.URX[j])+","+str(self.URY[j])+")\n"   
#                f.write(txt)
#                if (mode == 'v'): print(txt)
#                txt = "\n\n" 
#        f.close()
#
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
    dirs=["/Users/drago/Desktop/Current/test/inpt/CCX/ClusterLevel3/"]
#    dirs=["/Users/drago/Desktop/Current/test/inpt/SPC/"]
#    dirs=["/Users/drago/Desktop/Current/test/inpt/test/"]

    graph = Graph()
#    RunCosts = [0, 1, 2]
#    RunCosts = [3, 4, 5]
    RunCosts = [0, 1, 2, 3, 4, 5]

    for mydir in dirs:
        cluster_file = mydir + "cluster.rpt"
        connect_file = mydir + "connectivity.rpt"
        graph.ReadClusters(cluster_file, 14, 2)
        graph.ReadConnectivity(connect_file, 34, 2)
        graph.SplitComma()
        graph.BuildEdges() 
        graph.BuildEdgeWeight() 
        for run in RunCosts:
            graph.WriteMetisFile(run)
        for run in RunCosts:
            graph.GraphPartition(run)
        for run in RunCosts:
            graph.WritePartitionDirectives(run)
