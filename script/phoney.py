"""
PHONEY [Pronounce 'poney'], Partitioning of Hypergraph Obviously Not EasY

Note: - It is very important at the moment that the clusters have the same ID as their
        position in the Graph.clusters list.
      - All .out files should be in the same <dir> folder. However, the script will 
        look the parent folder if they can't be found.

Usage:
    phoney.py   [-d <dir>] [-w <weight>] [--seed=seed] [--algo=algo] [--path=path] [--fix-pins]
                [--simple-graph] [--custom-fixfile] [--netsegments] [--ub=UBfactor]
    phoney.py --help

Options:
    -d <dir>            Design directory containing the cluster information (.out files and such)
    -w <weight>         Amount of edge weights to consider, [1, 10]
    --seed=seed         RNG seed
    --algo=algo         0: hMETIS [default: 0]
                        1: PaToH
                        2: Circut
                        3: Custom partitioner
                        4: Random partitioner
    --path=path         Full path to the partitioning tool
    --simple-graph      If set, translate the hypergraph into a simple graph
    --fix-pins          If set, force all standard cells connected to a pin to be placed
                        on the bottom die.
    --custom-fixfile    Use existing fixfile for hmetis, located in <dir> and named 'fixfile.hgr'
    --netsegments       Use alternative net segments for the graph generation. Needs a file named 'WLnets_segments.out'.
    --ub=UBfactor       Hmetis' UBfactor. eg ub=1 means partitions allowed to a disbalance of 49/51
    -h --help           Print this help
"""

import math
from subprocess import call
import subprocess
import copy
import matplotlib.pyplot as plt
import threading
from multiprocessing import Process, Pipe
import time
import pickle
import random
# from sets import Set
import os
import datetime
import sys
import logging, logging.config
import shutil
from docopt import docopt
import statistics
from alive_progress import alive_bar

global HMETIS_PATH
HMETIS_PATH = "/home/para/dev/metis_unicorn/hmetis-1.5-linux/"
HMETIS_PATH_BETA = "/home/para/dev/metis_unicorn/hmetis-2.0pre1/Linux-x86_64/hmetis2.0pre1"
HMETIS_BETA = True
METIS_PATH = "/home/para/dev/metis_unicorn/metis/bin/"
PATOH_PATH = "/home/para/Downloads/patoh/build/Linux-x86_64/"
CIRCUT_PATH = "/home/para/dev/circut10612/circut_v1.0612/tests/"
PIN_CELLS_F = "pinCells.out"
PIN_COORD_F = "pinCoord.out"
ALGO = 0 # 0: METIS
         # 1: PaToH
         # 2: Circut
ALGO_DICO = {0: "hMetis", 1: "PaToH", 2: "Circut", 3: "arbitrary-part", 4: "random-part"}
EDGE_WEIGHTS_TYPES = 10
# EDGE_WEIGHTS_TYPES = 1
VERTEX_WEIGHTS_TYPES = 1
WEIGHT_COMBILI_COEF = 0.5
MAX_WEIGHT = 100000
THREADS = 1     # Amount of parallel process to use when building the hypergraph.
CLUSTER_INPUT_TYPE = 0  # 0: standard .out lists
                        # 1: Ken's output
                        # 2: Custom clustering, no boundaries
SIMPLE_GRAPH = False    # False: hypergraph, using hmetis
                        # True: standard graph, using gpmetis
MEMORY_BLOCKS = False   # True if there are memory blocks (bb.out)
IMPORT_HYPERGRAPH = False    # True: import the hypergraph from a previous dump,
                            # skip the graph building directly to the partitioning.
DUMP_FILE = 'hypergraph.dump'
# DUMP_FILE = 'simplegraph.dump'

HETEROGENEOUS_FACTOR = 5 #

RANDOM_SEED = 0 # 0: no seed, pick a new one
                # other: use this

DUMMY_CLUSTER = False # Add a dummy cluster (low area, high power)
DUMMY_NAME = "dummy_cluster"
POWER_ASYMMETRY = 50    # [50; 100]
                        # At 50: symmetry
                        # At 100: everything in one partition
MANUAL_ASYMMETRY = False # Execute a manual asymmetrisation of the power across partitions.

POWER_DENSITIES = [1.0, 0.6, 0.45, 0.42, 0.39, 0.22, 0.18, 0.11, 0.10, 0.08, 0.08, 0.05, 0.05]

# If True, each cluster contains only one gate.
# In that case, we assume there are no two identicals nets
# and there is no need to merge the corresponding hyperedges
ONE_TO_ONE = True
FIX_PINS = False

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


def buildHyperedges(startIndex, endIndex, nets, clusters):

    hyperedges = []
    logger.debug("We have %i nets and %i clusters that need to be converted into a graph.", len(nets), len(clusters))
    # print clusters
    i = startIndex
    while i < endIndex:
        net = nets[i]

        if len(net.clusters) > 1:
            if SIMPLE_GRAPH:
                # logger.warning("Simple graph EXPERIMENTAL")
                # logger.debug(i)
                clusters = list(net.clusters)
                for k in range(len(clusters)):
                    for j in range(k+1,len(clusters)):
                        hyperedge = Hyperedge()
                        hyperedge.addCluster(clusters[k])
                        hyperedge.addCluster(clusters[j])
                        hyperedge.addNet(net)
                        hyperedges.append(hyperedge)
            else:
                hyperedge = Hyperedge()
                for cluster in net.clusters:
                    hyperedge.addCluster(cluster)
                hyperedge.addNet(net)
                hyperedges.append(hyperedge)

        progression = printProgression(i - startIndex, endIndex - startIndex)
        if progression != "":
            logger.info(progression)
        i += 1

    return hyperedges

def closeEnough(target, value):
    tolerance = 0.2
    if value < (target + tolerance*target) and \
        value > (target - tolerance*target):
        return True
    else:
        return False

def elementIsInList(element, theList):
    found = False
    try:
        theList.index(element)
    except:
        pass
    else:
        found = True
    return found



class Graph():
    def __init__(self):
        self.clusters = dict() # dictionary of Cluster objects, key: cluster name
        self.nets = [] # list of Net objects.
        self.instances = dict() # dictionary of instances. Key: instance name
        self.hyperedges = []
        self.hyperedgeWeightsMax = []   # Maximum weight for each weight type.
                                        # Ordered by weight type.
        self.partitions = []    # Each element is a list of clusters (objects)
                                # included in the corresponding partition.
        self.partitionsArea = []
        self.partitionsPower = []
        self.logfilename = "graph.log"
        call(["rm","-rf","graph.log"])
        # TODO add a dictionary of instances. It will contain all the instances with their names as keys.
        # It will be populated the first time we extract the instances information (let's say with the clusters).
        # Then, when we extract the info about nets (if we began with clusters), for each instance encountered, we can look it up in the dictionary so that we can fetch its net object reference and add it to the net object.

    def WriteLog(self, obj):
        f = open(self.logfilename, "a")
        f.write(str(obj)+"\n")
        f.close()

    def ReadClusters(self, filename, hrows, frows):
        logger.info("Reading clusters file: %s", filename)
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        # Remove the header lines
        for i in range(0, hrows):
            del lines[0]
        # Remove the footer lines
        for i in range(0,frows):
            del lines[-1]

        if CLUSTER_INPUT_TYPE == 0:
            for i, line in enumerate(lines):
                line = line.strip(' \n')
                clusterDataRow = line.split()
                cluster = Cluster(clusterDataRow[0], i, False)

                lowerBounds = clusterDataRow[3][1:-1].split(",")    # Exclude the first and last
                                                                    # chars: '(' and ')'.
                upperBounds = clusterDataRow[4][1:-1].split(",")
                cluster.setBoundaries(float(lowerBounds[0]), float(lowerBounds[1]),
                    float(upperBounds[0]), float(upperBounds[1]))

                cluster.setArea(float(clusterDataRow[5]))
                self.clusters[cluster.name] = cluster
        elif CLUSTER_INPUT_TYPE == 1:
            for i, line in enumerate(lines):
                line = line.strip(' \n')
                clusterDataRow = line.split()
                cluster = Cluster(clusterDataRow[0], i, False)
                cluster.setArea(clusterDataRow[1])
                self.clusters[cluster.name] = cluster
        elif CLUSTER_INPUT_TYPE == 2:
            for i, line in enumerate(lines):
                line = line.strip(' \n')
                clusterDataRow = line.split()
                cluster = Cluster(clusterDataRow[0], i, False)

                cluster.setArea(float(clusterDataRow[4]))
                self.clusters[cluster.name] = cluster


    def readClustersInstances(self, filename, hrows, frows):
        # print "--------------------------------------------------->"
        logger.info("Reading clusters instances file: %s", filename)
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        if CLUSTER_INPUT_TYPE == 0:
            # Remove the header lines
            for i in range(0, hrows):
                del lines[0]
            # Remove the footer lines
            for i in range(0,frows):
                del lines[-1]

            for line in lines:
                line = line.strip(' \n')
                line = line.replace('{','')
                line = line.replace('}','')
                # print line
                clusterInstancesRow = line.split()
                try:
                    cluster = self.clusters[clusterInstancesRow[0]]
                except KeyError:
                    pass
                else:
                    del clusterInstancesRow[0]
                    for i, instanceName in enumerate(clusterInstancesRow):
                        instance = Instance(instanceName)
                        instance.addCluster(cluster)
                        self.instances[instanceName] = instance
                        cluster.addInstance(instance)

        elif CLUSTER_INPUT_TYPE == 1:
            # line sample : add_to_cluster -inst lb/rtlc_I2820 c603
            # TODO change this branch so that the clusters use Instance objects instead of simply their names
            for i, line in enumerate(lines):
                line = line.strip(' \n')
                clusterInstancesRow = line.split()
                try:
                    cluster = self.clusters[clusterInstancesRow[3]]
                except KeyError:
                    logger.warning("Das ist ein Problem, cluster %s not found.", clusterInstancesRow[3])
                else:
                    foundInstance = cluster.searchInstance(clusterInstancesRow[2])
                    if not foundInstance:
                        cluster.addInstance(clusterInstancesRow[2])

                progression = printProgression(i, len(lines))
                if progression != "":
                    logger.info(progression)

    def instancesCoordinates(self, filename, hrows, frows):

        logger.info("Reading instance coordinates file: {}".format(filename))
        try:
            with open(filename, 'r') as f:
                lines = f.read().splitlines()
        except IOError:
            with open(os.sep.join([os.sep.join(filename.split(os.sep)[:-2]), filename.split(os.sep)[-1]]), 'r') as f:
                lines = f.read().splitlines()

        coordinates = {} # {instance name : [x, y]}
        for l in lines:
            coordinates[l.split(',')[1]] = [float(l.split(',')[2]), float(l.split(',')[3])]

        for cluster in self.clusters.values():
            for instance in cluster.instances.values():
                instance.x = coordinates[instance.name][0]
                instance.y = coordinates[instance.name][1]




    def readMemoryBlocks(self, filename, hrows, frows):
        logger.info("Reading memory blocks file: %s", filename)
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        # Remove the header lines
        for i in range(0, hrows):
            del lines[0]
        # Remove the footer lines
        for i in range(0,frows):
            del lines[-1]

        for i in range(0, len(lines)):
            lines[i] = " ".join(lines[i].split())

        i = 0
        existingClustersCount = len(self.clusters) # Clusters already in the list before.
        blockCount = 0
        while i < len(lines):
            line = lines[i]
            #[0]: block name
            #[2]: instance count
            #[4]: physical area
            #[10]: instance
            blockRow = line.split()
            if len(line) > 0:
                if CLUSTER_INPUT_TYPE == 0:
                    memoryBlock = Cluster(blockRow[10], existingClustersCount + blockCount, True)
                    blockCount += 1
                    moduleArea = float(blockRow[4])
                    memoryBlock.setArea(moduleArea)
                    instance = Instance(blockRow[10])
                    instance.addCluster(memoryBlock) # When we create an Instance, we need to add the cluster it belongs to (needed by the nets for the buildHyperedge).
                    self.instances[instance.name] = instance
                    memoryBlock.addInstance(instance)
                    self.clusters[memoryBlock.name] = memoryBlock

                    for j in range(1, int(blockRow[2])):
                        i += 1
                        line = lines[i]
                        subBlockRow = line.split()
                        memoryBlock = Cluster(subBlockRow[1], existingClustersCount + blockCount, True)
                        blockCount += 1
                        memoryBlock.setArea(moduleArea)
                        instance = Instance(subBlockRow[1])
                        instance.addCluster(memoryBlock)
                        self.instances[instance.name] = instance
                        memoryBlock.addInstance(instance)
                        self.clusters[memoryBlock.name] = memoryBlock

                elif CLUSTER_INPUT_TYPE == 1:
                    # TODO 2017-07-20 this is probably broken since we changed the data structures inside the classes
                    # in commit 493fb6aa2aa12f8f681adfebe8a3c9c8b3d320db
                    moduleArea = float(blockRow[4])
                    for key in self.clusters:
                        cluster = self.clusters[key]
                        if cluster.searchInstance(blockRow[10]):
                            cluster.setArea(moduleArea)
                            cluster.blackbox = True
                            break
                        k += 1

                    for j in range(1, int(blockRow[2])):
                        i += 1
                        line = lines[i]
                        subBlockRow = line.split()
                        for key in self.clusters:
                            cluster = self.clusters[key]
                            if cluster.searchInstance(subBlockRow[1]):
                                cluster.setArea(moduleArea)
                                cluster.blackbox = True
                                break


            i += 1
            progression = printProgression(i, len(lines))
            if progression != "":
                logger.info(progression)


    def readNetsWireLength(self, filename, hrows, frows, segments):
        logger.info("Reading wire length nets file: %s", filename)
        try:
            with open(filename, 'r') as f:
                lines = f.read().splitlines()
        except IOError:
            with open(os.sep.join([os.sep.join(filename.split(os.sep)[:-2]), filename.split(os.sep)[-1]]), 'r') as f:
                lines = f.read().splitlines()

        # Remove the header lines
        for i in range(0, hrows):
            del lines[0]
        # Remove the footer lines
        for i in range(0,frows):
            del lines[-1]

        # if segments:
        #     lines.sort()
        #     for i, line in enumerate(lines):
        #         net = Net(line.split()[0], i)
        #         net.setPinAmount(int(line.split()[1]))
        #         net.setWL(float(line.split()[2]))
        #         self.nets.append(net)
        #         progression = printProgression(i, len(lines))
        #         if progression != "":
        #             logger.info(progression)

        # else:
        for i in range(0, len(lines)):
            lines[i] = " ".join(lines[i].split())
        lines.sort()

        with alive_bar(len(lines)) as bar:
            for i, line in enumerate(lines):
                netDataRow = line.split()

                net = Net(netDataRow[0], i)
                net.setPinAmount(int(netDataRow[1]))
                net.setWL(float(netDataRow[2]))
                # net.setWL(int(float(netDataRow[2])) + 1)    # + 1 here to make sure
                                                            #we don't use a net with WL = 0
                self.nets.append(net)
                # progression = printProgression(i, len(lines))
                # if progression != "":
                #     logger.info(progression)
                bar()


    def readNets(self, filename, hrows, frows, segments):
        logger.info("Reading nets file: %s", filename)
        try:
            with open(filename, 'r') as f:
                lines = f.read().splitlines()
        except IOError:
            with open(os.sep.join([os.sep.join(filename.split(os.sep)[:-2]), filename.split(os.sep)[-1]]), 'r') as f:
                lines = f.read().splitlines()

        # Remove the header lines
        for i in range(0, hrows):
            del lines[0]
        # Remove the footer lines
        for i in range(0,frows):
            del lines[-1]

        lines.sort()

        if segments:
            netID = 0
            i = 0
            with alive_bar(len(lines)) as bar:
                while i < len(lines) and netID < len(self.nets):
                    line = lines[i]
                    line = line.replace('{','')
                    line = line.replace('}','')
                    netDataRow = line.split()
                    if self.nets[netID].name == netDataRow[0]:
                        for instanceName in netDataRow[0].split('/')[1:]:
                            # This ought to be a pin. Continue, skip the rest of the loop.
                            # TODO but we should make sure it's a pin indeed.
                            if instanceName not in self.instances.keys():
                                # logger.error("'{}' is not a valid instance.\nNet line: '{}'".format(instanceName, line))
                                continue
                            instance = self.instances.get(instanceName)
                            if instance is None:
                                logger.info("%s is not recognized as an instance.", str(instanceName))
                            instance.addNet(self.nets[netID])
                            self.nets[netID].addInstance(instance)
                            self.nets[netID].addCluster(instance.cluster)
                            # TODO add cluster reference to net from the instance
                            # self.nets[netID].addInstance(instanceName)
                        netID += 1
                    i += 1

                    # progression = printProgression(i, len(lines))
                    # if progression != "":
                    #     logger.info(progression)
                    bar()
        else:
            # As the nets list is sorted, we should not run through all the
            # self.nets list for each line.
            # For each line, move forward in self.nets. If the net in the line
            # is found, great! If not, no biggie, just keep moving forward.
            netID = 0
            i = 0
            with alive_bar(len(lines)) as bar:
                while i < len(lines) and netID < len(self.nets):
                    line = lines[i]
                    line = line.strip(' \n')
                    line = line.replace('{','')
                    line = line.replace('}','')
                    netDataRow = line.split()
                    if self.nets[netID].name == netDataRow[0]:
                        del netDataRow[0]
                        for instanceName in netDataRow:
                            # TODO add a verification that the instance is indeed in the class instances dictionary.
                            instance = self.instances.get(instanceName)
                            if instance is None:
                                logger.info("%s is not recognized as an instance.", str(instanceName))
                            instance.addNet(self.nets[netID])
                            self.nets[netID].addInstance(instance)
                            self.nets[netID].addCluster(instance.cluster)
                            # TODO add cluster reference to net from the instance
                            # self.nets[netID].addInstance(instanceName)
                        netID += 1
                    i += 1

                    # progression = printProgression(i, len(lines))
                    # if progression != "":
                    #     logger.info(progression)
                    bar()


    def readPins(self, filename):
        """
        Pins are expected to share the same name with the connected net.

        For each pin, create a dummy cluster, tagged as a pin and with a null area.

        parameters:
        -----------
        filename : String
            Path to a file formated as follows:
            <Pin name> <x [float]> <y [float]>
        """
        logger.info("Reading  pins file: {}".format(filename))
        try:
            with open(filename, 'r') as f:
                lines = f.read().splitlines()
        except IOError:
            with open(os.sep.join([os.sep.join(filename.split(os.sep)[:-2]), filename.split(os.sep)[-1]]), 'r') as f:
                lines = f.read().splitlines()

        for line in lines:
            cluster = Cluster(line.split(' ')[0], len(self.clusters), False)
            cluster.isPin = True
            cluster.setArea(0)
            cluster.setPower(0)
            self.clusters[cluster.name] = cluster
            for net in self.nets:
                if net.name == cluster.name:
                    net.clusters.add(cluster)




##     ##  ##    ##   ########   #########  ########   #########  ######      #######   #########  
##     ##   ##  ##    ##     ##  ##         ##     ##  ##         ##    ##   ##         ##         
##     ##    ####     ##     ##  ##         ##     ##  ##         ##     ##  ##         ##         
#########     ##      #######    ######     ########   ######     ##     ##  ##   ####  ######     
##     ##     ##      ##         ##         ##   ##    ##         ##     ##  ##     ##  ##         
##     ##     ##      ##         ##         ##    ##   ##         ##    ##   ##     ##  ##         
##     ##     ##      ##         #########  ##     ##  #########  ######     ########   #########  


    def findHyperedges(self, outputDir):
        logger.info("Building hyperedges")

        self.hyperedges = buildHyperedges(0, len(self.nets), self.nets, self.clusters)

        s = ""
        for hyperedge in self.hyperedges:
            s += str(len(hyperedge.clusters)) + "\t" + str(hyperedge.nets[0].name)
            for cluster in hyperedge.clusters:
                s += " " + str(cluster.name)
            s += "\n"
        with open(os.path.join(outputDir, "raw_hyperedges.out"), 'w') as f:
            f.write(s)

        if not ONE_TO_ONE:

            logger.info("Merging hyperedges")
            '''
            Merging hyperedges is looking for hyperedges with the same amount of vertices,
            check if those are the same and if so, add the nets from the second hyperedge
            to the first.
            As each hyperedge needs to be compared to the n-i next ones, this part is at
            least of complexity O(nlogn).
            '''
            logger.info("We begin with %s hyperedges.", str(len(self.hyperedges)))
            i = 0
            duplicateCount = 0
            errorCount = 0
            dumpDuplicates = ""
            dumpUniques = ""
            while i < len(self.hyperedges):
                hyperedgeA = self.hyperedges[i]
                duplicate = False
                # TODO delete clusterAMerged. This is useless. Since we already compare A to _all_ subsequent hyperedges, there is no need to mark it as merged.
                clusterAMerged = False  # Flag set in case an hyperedgeA is merged
                                        # into an hyperedgeB.
                j = i + 1
                while j < len(self.hyperedges) and not clusterAMerged:
                    hyperedgeB = self.hyperedges[j]
                    if len(hyperedgeA.clusters) == len(hyperedgeB.clusters):

                        # Find duplicates
                        duplicate = True
                        for clusterA in hyperedgeA.clusters:
                            nameFound = False
                            for clusterB in hyperedgeB.clusters:
                                if clusterA.name == clusterB.name:
                                    nameFound = True
                                    break
                            if not nameFound:
                                duplicate = False
                                break


                        # Check if the found duplicates are correct.
                        # This can be used as debug.
                        # if duplicate:
                        #     # Check if the duplicate is correct
                        #     if len(hyperedgeA.clusters) != len(hyperedgeB.clusters):
                        #         print "False duplicate: different length."
                        #     else:
                        #         error = False
                        #         for clusterA in hyperedgeA.clusters:
                        #             nameFound = False
                        #             for clusterB in hyperedgeB.clusters:
                        #                 if clusterA.name == clusterB.name:
                        #                     nameFound = True
                        #             if not nameFound:
                        #                 error = True
                        #         if error:
                        #             errorCount += 1
                        #             print "False duplicate: different cluster name."
                        #             print "Cluster A: "
                        #             for cluster in hyperedgeA.clusters:
                        #                 print cluster.name
                        #             if clustersA != None:
                        #                 print "Set: " + str(clustersA)
                        #             print "Cluster B: "
                        #             for cluster in hyperedgeB.clusters:
                        #                 print cluster.name
                        #             if clustersB != None:
                        #                 print "Set: " + str(clustersB)
                        #             print "Python thought the difference was " + str(setsDifference) + " which it considered empty = " + str(bool(not setsDifference))

                        #     # Dump the duplicates
                        #     for cluster in hyperedgeA.clusters:
                        #         dumpDuplicates += str(cluster.name) + " "
                        #     dumpDuplicates += "\n"
                        #     for cluster in hyperedgeB.clusters:
                        #         dumpDuplicates += str(cluster.name) + " "
                        #     dumpDuplicates += "\n"


                        if duplicate:
                            # Append the net from hyperedgeB to hyperedgeA.
                            # At this point, hyperedgeB only has one edge,
                            # because it has not been merged with anything
                            # and only merged hyperedges can have more than
                            # one edge.
                            duplicateCount += 1
                            hyperedgeA.addNet(hyperedgeB.nets[0])
                            hyperedgeA.connectivity += 1
                            del self.hyperedges[j]
                            # clusterAMerged = True
                        else:
                            j += 1


                    else:
                        j += 1

                # If hyperedgeA has not been merged, inspect the next one.
                # Otherwise, hyperedgeB has been deleted and all the following
                # elements have been shifted, thus no need to increment the index.
                if not clusterAMerged:
                    i += 1

                progression = printProgression(i, len(self.hyperedges))
                if progression != "":
                    logger.info(progression)

            logger.info("Duplicates: %s", str(duplicateCount))
            logger.info("Errors: %s", str(errorCount))

        logger.info("We end up with %s hyperedges.", str(len(self.hyperedges)))

        # for hyperedge in self.hyperedges:
        #     for cluster in hyperedge.clusters:
        #         dumpUniques += str(cluster.name) + " "
        #     dumpUniques += "\n"

        # with open("duplicates.dump", 'w') as file:
        #     file.write(dumpDuplicates)
        # with open("non-duplicates.dump", 'w') as f:
        #     f.write(dumpUniques)

        if SIMPLE_GRAPH:
            logger.info("Prepare simple graph")
            for i, hyperedge in enumerate(self.hyperedges):
                for k, clusterA in enumerate(hyperedge.clusters):
                    for l, clusterB in enumerate(hyperedge.clusters):
                        if l != k:
                            clusterA.connectedClusters.append(clusterB.name)
                            clusterA.connectedEdges.append(hyperedge)


##     ##   #######   ########   
##     ##  ##         ##     ##  
##     ##  ##         ##     ##  
#########  ##   ####  ########   
##     ##  ##     ##  ##   ##    
##     ##  ##     ##  ##    ##   
##     ##  ########   ##     ##  



    def generateMetisInput(self, filename, edgeWeightType, vertexWeightType, fixPins, metisInputFixfile, pinCellsFile, CUSTOM_FIXFILE=False):
        """
        
        Parameters
        ----------
        fixPins : boolean
            If true, tell Metis to force the cells in PIN_CELLS_F to stay on the layer 0.
            Extract from the documentation:
            The FixFile is used to specify the vertices that are pre-assigned to certain partitions. In general, when computing a
            k-way partitioning, up to k sets of vertices can be specified, such that each set is pre-assigned to one of the k partitions.
            For a hypergraph with |V| vertices, the FixFile consists of |V| lines with a single number per line. The ith line of the
            file contains either the partition number to which the ith vertex is pre-assigned to, or -1 if that vertex can be assigned
            to any partition (i.e., free to move). Note that the partition numbers start from 0.

        CUSTOM_FIXFILE : boolean
            If true, don't regenarate the fixfile.
        """
        logger.info("Generating METIS input file...")
        s = ""
        if SIMPLE_GRAPH:
            s = str(len(self.clusters)) + " " + str(len(self.hyperedges)) + " 011"

            if VERTEX_WEIGHTS_TYPES > 1:
                s += " " + str(VERTEX_WEIGHTS_TYPES)

            for key in self.clusters:
                # print cluster
                cluster = self.clusters[key]
                s += "\n"
                for weightType in range(0, VERTEX_WEIGHTS_TYPES):
                    s += str(cluster.weightsNormalized[weightType]) + " "
                # for j in range(0, len(cluster.connectedClusters)):
                for j, clusterName in enumerate(cluster.connectedClusters):
                    s += " " + str(self.clusters[clusterName].ID + 1) + \
                        " " + str(cluster.connectedEdges[j].weightsNormalized[edgeWeightType])


        else:
            s = str(len(self.hyperedges)) + " " + str(len(self.clusters)) + " 11"
            for i, hyperedge in enumerate(self.hyperedges):
                s += "\n" + str(int(math.ceil(hyperedge.weightsNormalized[edgeWeightType]))) + " " # ceil to make sure it's > 0.
                for cluster in hyperedge.clusters:
                    s += str(cluster.ID + 1) + " "  # hmetis does not like to have hyperedges
                                                    # beginning with a cluster of ID '0'.
            for key in self.clusters:
                cluster = self.clusters[key]
                s += "\n" + str(cluster.weightsNormalized[vertexWeightType])
        with open(filename, 'w') as file:
            file.write(s)

        if not CUSTOM_FIXFILE:
            if fixPins:
                try:
                    with open(pinCellsFile, 'r') as f:
                        cellPins = f.read().splitlines()
                except IOError:
                    with open(os.sep.join([os.sep.join(pinCellsFile.split(os.sep)[:-2]), pinCellsFile.split(os.sep)[-1]]), 'r') as f:
                        cellPins = f.read().splitlines()

                s = ""
                # For each vertex, check in PIN_CELLS_F if a cell in the cluster is connected to a pin.
                # If it is, set its line to 0.
                # Set the line to -1 otherwise.
                for cluster in self.clusters.values():
                    gotPin = -1 # Does the cluster contain a cell connected to a pin?
                    for cell in cluster.instances.keys():
                        if cell in cellPins:
                            gotPin = 0
                            break
                    s += "{}\n".format(gotPin)
                with open(metisInputFixfile, 'w') as f:
                    f.write(s)
            else:
                s = ""
                for cluster in self.clusters.values():
                    if cluster.isPin or cluster.assignedBottom:
                        s += "0\n"
                    else:
                        s += "-1\n"
                with open(metisInputFixfile, 'w') as f:
                    f.write(s)



    def generatePaToHInput(self, filename, edgeWeightType, vertexWeightType):
        logger.info("Generating PaToH input file...")
        s = ""

        # Compute the amount of 'pins' in the hypergraph.
        # For PaToH, a pin is a vertex in a hyperedge.
        # So to compute this amount, we simply need to now how
        # many vertex (cluster, here) are in each hyperedge.
        pins = 0
        for hyperedge in self.hyperedges:
            pins += len(hyperedge.clusters)

        # TODO add the possibility to have all the vertex weights types in the same description (PaToH can do that).
        s = "1 " + str(len(self.clusters)) + " " + str(len(self.hyperedges)) + " " + str(pins) + " 3" + " 1"
        for i, hyperedge in enumerate(self.hyperedges):
            s += "\n" + str(hyperedge.weightsNormalized[edgeWeightType]) + " "
            for cluster in hyperedge.clusters:
                s += str(cluster.ID + 1) + " "

        s += "\n"

        for key in self.clusters:
            s += str(self.clusters[key].weightsNormalized[vertexWeightType]) + " "

        s += "\n" # Requires a new line at the end of the file.

        with open(filename, 'w') as file:
            file.write(s)


    def generateCircutInput(self, filename, edgeWeightType, vertexWeightType):
        logger.info("Generating Circut input file...")
        s = str(len(self.clusters)) + " " + str(len(self.hyperedges)) + "\n"


        for h in self.hyperedges:
            for c in h.clusters:
                s += " " + str(c.ID + 1)
            s += " " + str(h.weightsNormalized[edgeWeightType])
            s += "\n"

        with open(filename, 'w') as file:
            file.write(s)


    def generateCustPartInput(self, filename, edgeWeightType, vertexWeightType):
        logger.info("Generating Custom Partitioner input file...")

        # First find the design width by finding the rightmost cluster boundary
        width = 0
        for ck in self.clusters:
            cluster = self.clusters[ck]
            if cluster.boundaries[1][1] > width:
                width = cluster.boundaries[1][1]

        for ck in self.clusters:
            cluster = self.clusters[ck]
            cluster.computeDistanceFromOrigin(width)

        clusterList = list(self.clusters.values())
        clusterList.sort(key=lambda x:x.distOr)

        s = ""
        for c in clusterList:
            s += str(c.ID) + "\n"
            # s += str(c.ID) + " (" + str(c.boundaries[0][0]) + ", " + str(c.boundaries[0][1]) + ")\n"

        with open(filename, 'w') as file:
            file.write(s)



##           ##  #########  ########    #######   ##     ##  ##########   #######   
##           ##  ##            ##      ##         ##     ##      ##      ##     ##  
##           ##  ##            ##      ##         ##     ##      ##      ##         
##           ##  ######        ##      ##   ####  #########      ##       #######   
 ##   ###   ##   ##            ##      ##     ##  ##     ##      ##             ##  
  ## ## ## ##    ##            ##      ##     ##  ##     ##      ##      ##     ##  
   ###   ###     #########  ########   ########   ##     ##      ##       ####### 

    def computeHyperedgeWeights(self, normalized):
        logger.info("Generating weights of hyperedges.")

        self.hyperedgeWeightsMax = [0] * EDGE_WEIGHTS_TYPES
        for weightType in range(0, EDGE_WEIGHTS_TYPES):
            for i, hyperedge in enumerate(self.hyperedges):
                weight = 0
                if weightType == 0:
                    # Number of wires
                    weight = hyperedge.connectivity

                    # ###
                    # # Experimental
                    # #
                    # pins = list()
                    # wl = 0
                    # for n in hyperedge.nets:
                    #     pins.append(n.pins)
                    #     wl += n.wl
                    # weight = statistics.mean(pins) / wl
                    # #
                    # ###

                    ###
                    # Another experimental
                    
                    # if SIMPLE_GRAPH:
                    #     manDist = [] # Manhattan distance of 'nets' in the simple graph edge
                    #     for cluster in hyperedge.clusters:
                    #         instances = list(cluster.instances.values())
                    #         manDist.append(abs(instances[0].x - instances[1].x) + abs(instances[0].y - instances[1].y))
                    #     weight = 1.0/statistics.mean(manDist)

                    #
                    ###

                    ###
                    # Yet another experimental
                    
                    # weight = random.uniform(1,100)

                    #
                    ###
                elif weightType == 1:
                    # 1/#wires
                    weight = 1.0 / hyperedge.weights[0]
                elif weightType == 2:
                    # Total wire length
                    for net in hyperedge.nets:
                        weight += net.wl
                elif weightType == 3:
                    # 1/Total wire length
                    weight = 1.0 / hyperedge.weights[2]
                elif weightType == 4:
                    # Average wire length
                    wlTot = 0
                    for net in hyperedge.nets:
                        wlTot += net.wl
                    wlAvg = wlTot / len(hyperedge.nets)
                    weight = wlAvg
                elif weightType == 5:
                    # 1/Average wire length
                    weight = 1.0 / hyperedge.weights[4]
                elif weightType == 6:
                    # Number of wires * total length
                    weight = hyperedge.weights[0] * hyperedge.weights[2]
                elif weightType == 7:
                    weight = 1.0 / hyperedge.weights[6]
                elif weightType == 8:
                    # total wire length and number of wires
                    weight = \
                        WEIGHT_COMBILI_COEF * hyperedge.weightsNormalized[0] + \
                        (1 - WEIGHT_COMBILI_COEF) * hyperedge.weightsNormalized[1]
                elif weightType == 9:
                    # 1 / total wire length and number of wires
                    weight = 1 / hyperedge.weights[8]
                elif weightType == 10:
                    # fanout
                    pins = 0
                    for n in hyperedge.nets:
                        pins += n.pins
                    weight = pins


                hyperedge.setWeight(weightType, weight)
                # Save the max
                if weight > self.hyperedgeWeightsMax[weightType]:
                    self.hyperedgeWeightsMax[weightType] = weight

            # Normalize
            for hyperedge in self.hyperedges:
                hyperedge.setWeightNormalized(weightType, int(((hyperedge.weights[weightType] * MAX_WEIGHT) / self.hyperedgeWeightsMax[weightType])) + 1)




    # def findSmallestCluster(self, clusters):
    #     smallest = clusters[0].area
    #     smallestID = 0
    #     for i, cluster in enumerate(clusters):
    #         if cluster.area < smallest:
    #             smallest = cluster.area
    #             smallestID = i

    #     return smallestID

    def computeClustersPower(self):
        '''
        For each cluster, set a random power density.
        The power of the cluster is thus its density * its area.
        '''
        for key in self.clusters:
            cluster = self.clusters[key]
            powerDensity = random.uniform(1, HETEROGENEOUS_FACTOR)
            power = cluster.area * powerDensity
            cluster.setPowerDensity(powerDensity)
            cluster.setPower(power)
            # print "Power density = " + str(powerDensity) + \
            #     ", area = " + str(cluster.area) + \
            #     ", power = " + str(power)
            cluster.setPowerDensity(powerDensity)
            cluster.setPower(power)


    def computeVertexWeights(self):
        logger.info("Generating weights of vertices.")
        self.clusterWeightsMax = [0] * VERTEX_WEIGHTS_TYPES
        totalPower = 0
        totalNormalizedPower = 0

        for weightType in range(0, VERTEX_WEIGHTS_TYPES):
            # Weight = cluster area
            if weightType == 0:
                for key in self.clusters:
                    cluster = self.clusters[key]
                    weight = cluster.area

                    if weight > self.clusterWeightsMax[weightType]:
                        self.clusterWeightsMax[weightType] = weight

                    cluster.setWeight(weightType, weight)
                    # print str(weight)

            # Weight = cluster power
            elif weightType == 1:

                for key in self.clusters:
                    cluster = self.clusters[key]
                    weight = cluster.power
                    totalPower += weight
                    cluster.setWeight(weightType, weight)

                    if weight > self.clusterWeightsMax[weightType]:
                        self.clusterWeightsMax[weightType] = weight

        # Normalization
        for key in self.clusters:
            cluster = self.clusters[key]
            for weightType in range(0, VERTEX_WEIGHTS_TYPES):
                weight = ((cluster.weights[weightType] * MAX_WEIGHT) / self.clusterWeightsMax[weightType]) + 1
                cluster.setWeightNormalized(weightType, int(weight))
                if weightType == 1:
                    totalNormalizedPower += cluster.weightsNormalized[weightType]
            # print cluster.weightsNormalized


        if DUMMY_CLUSTER:
            # Append the dummy cluster to the cluster list.
            dummy = Cluster(DUMMY_NAME, len(self.clusters), False)
            # If we want 70/30 asymmetry, the dummy power will be
            # totalNormalizedPower * 0.4, using 40 % of one partition's power
            dummyPower = totalNormalizedPower * (POWER_ASYMMETRY - (100 - POWER_ASYMMETRY)) / 100
            logger.info("totalNormalizedPower: %s, dummyPower: %s", 
                str(totalNormalizedPower), str(dummyPower))
            dummy.setArea(0.1)
            dummy.setPower(dummyPower)
            dummy.setWeight(0, 0.1)
            dummy.setWeight(1, dummy.power)
            dummy.setWeightNormalized(0, 1)
            dummy.setWeightNormalized(1, dummy.power)
            dummy.setDummy(True)
            self.clusters[dummy.name] = dummy



########      ###     ########   ##########  
##     ##    ## ##    ##     ##      ##      
##     ##   ##   ##   ##     ##      ##      
#######    ##     ##  ########       ##      
##         #########  ##   ##        ##      
##         ##     ##  ##    ##       ##      
##         ##     ##  ##     ##      ##  

    def GraphPartition(self, filename, FIX_PINS, fixfilepath, UBfactor):
        '''
        Call the hmetis command with the given filename, or gpmetis if we are working
        with simple graphs.
        '''

        logger.info("--------------------------------------------------->")
        # print "Running partition on cost: ", CostFunction
        logger.info("Running hmetis with %s", filename)
        # call(["/Users/drago/bin/hmetis-1.5-osx-i686/hmetis",filename,"2","5","20","1","1","1","0","0"])
        # hmetis graphFile Nparts UBfactor Nruns Ctype Rtype Vcycle Reconst dbglvl
        Nparts = 2
        # Nruns = 20
        Nruns = 10 # default for shmetis
        Ctype = 1
        Rtype = 1
        # Vcycle = 3 # Refinement at each step.
        Vcycle = 1 # Refinement only of the final bisection step.
        Reconst = 0
        dbglvl = 8
        command = ""
        fixfile = " {}".format(fixfilepath)
        # fixfile = " /home/para/dev/def_parser/2020-07-30_10-57-03_spc-2020_OneToOne/spc_Memory-on-logic_pur/metis_01_NoWires_area_fixfile.hgr"
        # logger.info("FIX_PINS is True, adding a fixfile located at {}".format(fixfile))
        if SIMPLE_GRAPH:
            command = METIS_PATH + "gpmetis " + filename + " 2 -dbglvl=0 -ufactor=30"
        else:
            if not HMETIS_BETA:
                command = HMETIS_PATH + "hmetis " + filename + fixfile + " " + str(Nparts) + \
                    " " + str(UBfactor) + " " + str(Nruns) + " " + str(Ctype) + " " + \
                    str(Rtype) + " " + str(Vcycle) + " " + str(Reconst) + " " + str(dbglvl)
                logger.info("Calling '%s'", command)
            else:
                logger.info("!!! BETA version of hMetis! !!!")
                command = "{} {} {} -ctype={} -rtype={} -ufactor={} -nruns={} -fixed={} -dbglvl={} -nvcycles={}".format(HMETIS_PATH_BETA, filename, Nparts, "fc1", "slow", UBfactor, Nruns, fixfilepath, dbglvl, Vcycle)
                logger.info("Calling {}".format(command))
        # call([HMETIS_PATH + "hmetis",filename,"2","5","20","1","1","1","0","0"])
        call(command.split())


    def GraphPartitionPaToH(self, filename):
        logger.info("--------------------------------------------------->")
        logger.info("Running PaToH with %s", filename)
        command = PATOH_PATH + "patoh " + filename + " 2 RA=6 UM=U OD=3"
        call(command.split())


    def GraphPartitionCircut(self, filename, outFilename):
        logger.info("--------------------------------------------------->")
        logger.info("Running Circut with %s", filename)
        workingDir = os.getcwd()
        tempWorkingDir = CIRCUT_PATH
        # TODO Change 'tmp.gset' so that I can run several instances.
        tempFile = os.path.join(CIRCUT_PATH, "tmp.gset")
        distantFilename = os.path.join(CIRCUT_PATH, filename.split(os.sep)[-1])
        distantOutFilename = os.path.join(CIRCUT_PATH, outFilename)
        shutil.copyfile(filename, distantFilename)
        with open(tempFile, 'w') as f:
            f.write(filename.split(os.sep)[-1] + "\n")
        command = "./circut < tmp.gset"
        print(command)
        os.chdir(tempWorkingDir)
        # TODO find a fracking way to get the output of the called command in the calling script. I want it in the log.
        child = subprocess.Popen(command, shell=True, stdout=None)
        child.wait()
        os.chdir(workingDir)
        # Copy back the partition file
        shutil.copyfile(distantOutFilename, os.path.join(os.sep.join(filename.split(os.sep)[:-1]), outFilename))
        os.remove(tempFile)
        os.remove(distantFilename)
        os.remove(distantOutFilename)

    def GraphPartitionCustPart(self, filename, outFilename):
        logger.info("Partitioning like a boss")
        logger.info("Running custom partitioner with {}".format(filename))


        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        part = dict() # {cluster ID, partition 0 or 1}

        for i, line in enumerate(lines):
            if i%2 == 0:
                part[int((line.strip()))] = 0
            else:
                part[int((line.strip()))] = 1

        s = ""
        for k in part:
            s += str(part[k]) + "\n"

        with open(outFilename, 'w') as f:
            f.write(s)

    def GraphPartitionRanPart(self, outFilename, write_output=True):
        '''
        Randomly put each cluster in a partition
        '''

        logger.info("Partitioning randomly")

        s = ""

        for ck in self.clusters:
            s += str(int(random.uniform(0,2))) + "\n" # 0 or 1

        if write_output:
            with open(outFilename, 'w') as f:
                f.write(s)
        else:
            return s




######     ########   ########   #########   #######   ##########  ########   ##     ##  #########   #######   
##    ##      ##      ##     ##  ##         ##     ##      ##         ##      ##     ##  ##         ##     ##  
##     ##     ##      ##     ##  ##         ##             ##         ##      ##     ##  ##         ##         
##     ##     ##      ########   ######     ##             ##         ##      ##     ##  ######      #######   
##     ##     ##      ##   ##    ##         ##             ##         ##       ##   ##   ##                ##  
##    ##      ##      ##    ##   ##         ##     ##      ##         ##        ## ##    ##         ##     ##  
######     ########   ##     ##  #########   #######       ##      ########      ###     #########   #######   



    def WritePartitionDirectives(self, metisFileIn, metisFileOut, gatePerDieFile, write_output=True, partStr=None):
        '''
        Create the partition directives in .tcl files.
        This is the final output telling which module (cluster) is going on which die.
        '''
        # print "--------------------------------------------------->"


        partFileStr = ""
        gpdStr = ""

        if write_output:
            logger.info("Write tcl file for file: %s", metisFileIn)
            try:
                fOut = open(metisFileOut, "w")
            except IOError:
                logger.error("Can't open file {}".format(metisFileOut))
                return False
            try:
                fIn = open(metisFileIn, "r")
            except IOError:
                logger.error("Can't open file {}".format(metisFileIn))
                return False
            try:
                fOutGates = open(gatePerDieFile, 'w')
            except IOError:
                logger.error("Can't open file {}".format(gatePerDieFile))
                return False

        # Find the longest cluster name first.
        # This is necessary in order to align the 'Diex' part
        # of the .tcl directives, allowing to apply easy column
        # edits later on.
        maxLength = 0
        for key in self.clusters:
            cluster = self.clusters[key]
            if len(cluster.name) > maxLength:
                maxLength = len(cluster.name)

        if write_output:
            data = fIn.readlines()
        else:
            data = partStr.split('\n')

        if ALGO==1: # In PaToH, everything is on one or more lines.
            tmpData = list()
            for l in range(len(data)):
                tmpData.extend(data[l].strip().split(' '))
            data = tmpData

        if ALGO==2: # Circut, die is -1 or 1. Change all -1 to 0.
            del data[0] # Remove the first header line
            for i in range(len(data)):
                data[i] = data[i].replace("-1", "0")
                data[i] = data[i].replace(" 1", "1") # remove the space so that we get 'Die1' and not 'Die 1'

        # print data

        # This enumeration assumes that the self.clusters dictionary
        # is sorted based on the cluster ID. However, as we use the
        # cluster name as a key, it is not the case.
        # Hence, we need a clone dictionary with the ID as key.
        # I think this the most efficient way to proceed time-wise.
        clusterDictID = dict()
        for key in self.clusters:
            clusterDictID[self.clusters[key].ID] = self.clusters[key]

        for i, key in enumerate(clusterDictID):
            cluster = clusterDictID[key]
            s = ""
            sInst = ""
            # Comment the dummy cluster out.
            if cluster.name == DUMMY_NAME:
                s += "#"
                cluster.setPartition(0)
            if cluster.blackbox:
                s += "add_to_die -inst    " + str(cluster.name) + \
                    " " * (maxLength - len(cluster.name)) + \
                    " -die Die" + str(data[i])
            else:
                s += "add_to_die -cluster " + str(cluster.name) + \
                    " " * (maxLength - len(cluster.name)) + \
                    " -die Die" + str(data[i])

            cluster.setPartition(data[i])

            if ALGO==1:
                s+= "\n"

            for instance in cluster.instances.keys():
                sInst += str(instance) + " " + str(data[i].strip()) + "\n"

            if write_output:
                fOut.write(s)   
                fOutGates.write(sInst)
            else:
                partFileStr += s + "\n"
                gpdStr += sInst
        if write_output:
            fOut.close()
            fIn.close()
            fOutGates.close()
        else:
            return partFileStr, gpdStr
        logger.info("Done!")
        # print "<---------------------------------------------------\n"

    def extractGraphNets(self):
        totNets = 0
        for hyperedge in self.hyperedges:
            totNets += len(hyperedge.nets)
        return totNets


    def extractPartitionConnectivity(self, conFile, weigthType, write_output=True):
        partCon = 0 # Connectivity across the partition
        currentPart = 0 # partition of the current hyperedge
        netCutStr = ""
        graphTotNets = self.extractGraphNets()
        fileStr = ""

        for hyperedge in self.hyperedges:
            for i, cluster in enumerate(hyperedge.clusters):
                if i == 0:
                    currentPart = cluster.partition
                else:
                    if currentPart == cluster.partition:
                        currentPart == cluster.partition
                    else:
                        partCon += hyperedge.connectivity
                        for net in hyperedge.nets:
                            netCutStr += "," + net.name
                        break

        logger.info("------------- Number of nets cut by the partitioning: %s out of %s -------------", 
            str(partCon), str(graphTotNets))
        fileStr = str(len(self.clusters)) + " clusters, " + str(graphTotNets) + " graphTotNets, "  + weigthType + " " + str(partCon) + " " + str(netCutStr) + "\n"
        if write_output:
            with open(conFile, 'a') as f:
                f.write(fileStr)
            return partCon
        else:
            return partCon, fileStr


    def extractTotalWL(self):
        '''
        Extract the total WL of the graph, not of the design.
        The difference between them resides in the intra-cluster nets
        that do not appear in the hypergraph.
        '''

        totLen = 0
        for hyperedge in self.hyperedges:
            totLen += hyperedge.getTotNetLength()
        return totLen



    def extractPartitionNetLengthCut(self, cutFile, weigthType, write_output=True):
        '''
        Extract the total length of all net cut by the partitioning and
        write it into "cutFile".

        This is done by iterating over all hyperedges.
        for each hyperedge, we check if all its clusters are contained inside a
        single partition. If not, we add the length of all the nets to the cut length.
        Indeed, as soon as a node of the hyperedge is separated, it means the hyperedge
        (and all the nets composing it) is cut.
        '''
        totLen = 0
        currentPart = 0
        graphTotLen = self.extractTotalWL()
        fileStr = ""

        for hyperedge in self.hyperedges:
            for i, cluster in enumerate(hyperedge.clusters):
                if i == 0:
                    currentPart = cluster.partition
                else:
                    if currentPart != cluster.partition:
                        totLen += hyperedge.getTotNetLength()
                        break
        logger.info("><><><><><><><>< total net length cut by the partitioning: %s out of %s", str(totLen), str(graphTotLen))

        fileStr = str(len(self.clusters)) + " clusters, " + str(graphTotLen) + " graphTotLen, " + weigthType + " " + str(totLen) + "\n"
        
        if write_output:
            with open(cutFile, 'a') as f:
                f.write(fileStr)
        else:
            return fileStr




    def dumpClusters(self):
        s = "ID -- Cluster -- Area -- Power density -- Power"
        # print "len(self.clusters) = " + str(len(self.clusters))
        for key in self.clusters:
            cluster = self.clusters[key]
            s += str(cluster.ID) + "\t" + cluster.name + "\t" + str(cluster.area) + "\t" + \
                str(cluster.powerDensity) + "\t" + str(cluster.power) + "\n"
        with open("clusters", 'w') as f:
            f.write(s)

    def hammingReport(self, filenames):
        '''
        As the output for a bipartition is either 0 or 1 (depending on which die a cluster is
        assigned to), by comparing the output of two different partition (e.g. based on different
        weights) for the same clusters, we can compute some sort of Hamming distance between
        them.
        This metric caracterize the difference bewteen two bipartitions.
        '''
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
                for i in range(0, len(lines)):
                    lines[i] = lines[i].strip(' \n')
                partitions.append(lines)
                # print lines

        table += "\n"

        for i in range(0, len(partitions) - 1): # ommit the last line, it has already been done as the last column
            table += str(i) + "\t" # Row name
            table += "\t" * i
            for j in range(i + 1, len(partitions)):
                hammingDistance = 0
                for bit in range(0, len(partitions[i])):
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
        logger.info(table)



    def plotWeights(self):
        # weights = [0] * EDGE_WEIGHTS_TYPES
        weights = []
        styles = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        for i in range(0, EDGE_WEIGHTS_TYPES):
            weights.append(list())
        for hyperedge in self.hyperedges:
            for i in range (0, EDGE_WEIGHTS_TYPES):
                weights[i].append(hyperedge.weightsNormalized[i])
        logger.info(weights[0])
        for i in range(0, EDGE_WEIGHTS_TYPES):
            plt.plot(weights[i], styles[i])
        plt.show()


    def computePartitionArea(self):

        self.partitionsArea = [0] * 2

        for i, partition in enumerate(self.partitions):
            # logger.debug("partition#: {}".format(i))
            for cluster in partition:
                self.partitionsArea[i] += cluster.area
                # logger.debug("cluster name: {}".format(cluster.name))

        totalArea = 0
        for part in self.partitionsArea:
            logger.debug("partition area: {}".format(part))
            totalArea += part

        areaFraction = [0] * 2
        for i, part in enumerate(self.partitionsArea):
            areaFraction[i] = float(part) / totalArea

        logger.info("Area: %s %s", str(self.partitionsArea), str(areaFraction))


    def computePartitionPower(self):

        self.partitionsPower = [0] * 2

        for i, partition in enumerate(self.partitions):
            for cluster in partition:
                self.partitionsPower[i] += cluster.power

        totalPower = 0
        for part in self.partitionsPower:
            totalPower += part

        powerFraction = [0] * 2
        if totalPower > 0:
            for i, part in enumerate(self.partitionsPower):
                powerFraction[i] = float(part) / totalPower


        logger.info("Power: %s %s", str(self.partitionsPower), str(powerFraction))


    def extractPartitions(self, partitionFile, readFile=True, partStr=None):
        logger.info("Extracting partitions directives.")

        if readFile:
            with open(partitionFile, 'r') as f:
                lines = f.read().splitlines()
        else:
            lines = partStr.split('\n')

        # reset partitions
        self.partitions = list()
        for i in range(0, 2):
            self.partitions.append(list())

        for line in lines:
            lineData = line.split()
            if len(lineData) > 0:
                if lineData[2] != DUMMY_NAME:
                    try:
                        cluster = self.clusters[lineData[2]]
                    except KeyError:
                        pass
                    else:
                        partitionIndex = int(lineData[4][3]) # Fourth character of 'DieX'
                        self.partitions[partitionIndex].append(cluster)



    def printPartitions(self):
        for i, partition in enumerate(self.partitions):
            logger.info("Partition #%s", str(i))
            for cluster in partition:
                logger.info("%s, %s, %s", cluster.name, str(cluster.area), str(cluster.powerDensity))


    def findHighestDensityCluster(self, clusters, unmatchableClusters):
        """
        return: - The power density of the selected cluster.
                - The ID of the cluster in the 'clusters' list.
        """
        den = 0 # highest density
        denCluster = None
        for i, cluster in enumerate(clusters):
            if cluster.powerDensity > den and \
            not elementIsInList(cluster.ID, unmatchableClusters) :
                den = cluster.powerDensity
                denCluster = i
        return denCluster, den

    def findLowDensityCluster(self, clusters, maxDensity, area):
        """
        - maxDensity is the density of the cluster that will replace the ones
            we pick here, and its power density should not be exceeded by them.
        - area is the area of the cluster to replace. The total area of the
            picked clusters here should be equal to that value.
        return: - a list of selected clusters (ID in the 'clusters' list)
        """
        clusterSelection = []
        areaSelection = 0 # total area of the selection


        # First, try to find one of the same are (+- 5 %) with a lower density
        # TODO: If several of the same area, keep the lowest density
        # for i, cluster in enumerate(clusters):
        #     isCloseEnough = closeEnough(area, cluster.area)
        #     if isCloseEnough and cluster.powerDensity < maxDensity:
        #         clusterSelection.append(i)
        #         areaSelection = cluster.area

        # If it does not work, use the accumulation of smaller clusters.
        # if len(clusterSelection) == 0:
        logger.info("Looking for low density clusters")
        # print "areaSelection: " + str(areaSelection) + ", target area: " + str(area)
        for i, cluster in enumerate(clusters):
            # print str(cluster.powerDensity) + " " + str(cluster.area)
            if cluster.powerDensity < maxDensity and \
            closeEnough(area, areaSelection + cluster.area) and \
            not elementIsInList(i, clusterSelection):
                areaSelection += cluster.area
                clusterSelection.append(i)

        return clusterSelection, areaSelection



    def manualOverride(self):
        """
        Use the self.partitions and move clusters around.
        Choose arbitrarily the partition 0 to host the low power part.
        The objective is to move clusters with high power away from this partition,
        whilst keeping the area balanced between the partitions.
        In partition 0, pick the cluster with the highest power density. Then, in
        partition 1, take the clusters with the lowest power density with the
        same total area as the cluster from partition 0.
        At the end of each iteration, we will evaluate the quality of the solution
        on two criteria: area balance and power asymmetry. If the asymmetry
        does not reach the target (e.g. 70/30), we keep going (with a limit
        to the number of iterations).
        """

        unmatchableClusters = [] # List of cluster IDs for which lower density
                                # cluster could not be found in partition 1.

        # for i in range(0, 10):
        #     print "RUN #" + str(i)
        #     highDenCluster, highDen = self.findHighestDensityCluster(self.partitions[0], unmatchableClusters)
        #     highDenArea = self.partitions[0][highDenCluster].area
        #     print "High density cluster: " + str(highDen) + ", area: " + str(highDenArea)
        #     # TODO: Check if highDenCluster is not none.

        self.printPartitions()

        i = 0
        while i < len(self.partitions[0]):
            highDenCluster = i
            highDen = self.partitions[0][highDenCluster].powerDensity
            highDenArea = self.partitions[0][highDenCluster].area

            lowDenClusters, lowDenArea = self.findLowDensityCluster(self.partitions[1], highDen, highDenArea)
            if closeEnough(highDenArea, lowDenArea):
                lowDenClusters.sort()
                logger.info("Low density clusters: (total area: %s)", str(lowDenArea))
                for id in lowDenClusters:
                    cluster = self.partitions[1][id]
                    logger.info("den: %s, area: %s", str(cluster.powerDensity), str(cluster.area))

                # Swap the clusters
                logger.info("BEFORE SWAPING:")
                for i, part in enumerate(self.partitions):
                    logger.info("Partition %s", str(i))
                    logger.info(part)
                    for c in part:
                        logger.info(c.ID)

                self.partitions[1].append(self.partitions[0][highDenCluster])
                del self.partitions[0][highDenCluster]
                for id in lowDenClusters:
                    self.partitions[0].append(self.partitions[1][id])

                # Delete in reversed order because when you delete an element, the
                # others are shifted to take its place.
                for id in reversed(lowDenClusters):
                    logger.info(str(id))
                    del self.partitions[1][id]

                logger.info("AFTER SWAPING:")
                for i, part in enumerate(self.partitions):
                    logger.info("Partition %s", str(i))
                    for c in part:
                        logger.info(c.ID)

            # If the area is not close enough, the high density cluster could
            # not be matched. Blacklist its ID.
            else:
                logger.info("High density cluster unmatchable")
                unmatchableClusters.append(self.partitions[0][highDenCluster].ID)
                i += 1

            self.computePartitionArea()
            self.computePartitionPower()
        self.printPartitions()







        ##      ##  #########  ##########  
        ###     ##  ##             ##      
        ## ##   ##  ##             ##      
        ##  ##  ##  ######         ##      
        ##   ## ##  ##             ##      
        ##     ###  ##             ##      
####### ##      ##  #########      ##      


class Net:
    def __init__(self, name, netID):
        self.name = name
        self.ID = netID
        self.wl = 0.0 # wire length
        self.instances = dict() # dictionary of instances. Key: instance name
        self.pins = 0 # number of pins
        self.clusters = set() # set of clusters. Having a Set is an advantage because it only containts unique objects.

    def setPinAmount(self, pins):
        self.pins = pins

    def setWL(self, wl):
        # if wl == 0 and self.pins > 1:
        #     print "#### STAP WL = 0: " + str(self.name)
        self.wl = wl

    def addInstance(self, instance):
        self.instances[instance.name] = instance
        # self.instances.append(instance)

    def searchInstance(self, instance):
        found = False
        try:
            self.instances.index(instance)
        except:
            pass
        else:
            found = True
        return found

    def addCluster(self, cluster):
        self.clusters.add(cluster)



         #######   ##         ##     ##   #######   ##########  #########  ########   
        ##     ##  ##         ##     ##  ##     ##      ##      ##         ##     ##  
        ##         ##         ##     ##  ##             ##      ##         ##     ##  
        ##         ##         ##     ##   #######       ##      ######     ########   
        ##         ##         ##     ##         ##      ##      ##         ##   ##    
        ##     ##  ##         ##     ##  ##     ##      ##      ##         ##    ##   
#######  #######   #########   #######    #######       ##      #########  ##     ##  

class Cluster:
    def __init__(self, name, clusterID, blackbox):
        self.name = name
        self.ID = clusterID
        self.instances = dict() # dictionary of instances. Key: instance name
        self.boundaries = [[0, 0], [0 ,0]] # [[lower X, lower Y], [upper X, upper Y]] (floats)
        self.area = 0 # float
        self.weights = []   # [0] = area
                            # [1] = power
                            # [2] = area & power
        self.weightsNormalized = []
        self.connectedClusters = [] # List of Cluster names connected to this cluster.
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
        self.power = 0
        self.powerDensity = 0
        self.isDummy = False
        self.partition = 0 # Partition to which the cluster belongs.
        self.distOr = 0 # Distance from the origin if all rows were aligned next to each other.
        self.isPin = False # Is nothing more than a pin.
        self.assignedBottom = False # If True, should be fixed on the bottom die.

    def setBoundaries(self, lowerX, lowerY, upperX, upperY):
        self.boundaries[0][0] = lowerX
        self.boundaries[0][1] = lowerY
        self.boundaries[1][0] = upperX
        self.boundaries[1][1] = upperY

    def addInstance(self, instance):
        """
        instance: Instance object
        """
        self.instances[instance.name] = instance

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

    def setPower(self, power):
        self.power = power

    def setPowerDensity(self, powerDensity):
        self.powerDensity = powerDensity

    def setDummy(self, status):
        self.isDummy = status

    def setPartition(self, part):
        self.partition = part

    def computeDistanceFromOrigin(self, width):
        self.distOr = self.boundaries[0][0] + self.boundaries[0][1] * width





        ##     ##  ##    ##   ########   #########  ########   #########  ######      #######   #########  
        ##     ##   ##  ##    ##     ##  ##         ##     ##  ##         ##    ##   ##         ##         
        ##     ##    ####     ##     ##  ##         ##     ##  ##         ##     ##  ##         ##         
        #########     ##      #######    ######     ########   ######     ##     ##  ##   ####  ######     
        ##     ##     ##      ##         ##         ##   ##    ##         ##     ##  ##     ##  ##         
        ##     ##     ##      ##         ##         ##    ##   ##         ##    ##   ##     ##  ##         
####### ##     ##     ##      ##         #########  ##     ##  #########  ######     ########   #########  


class Hyperedge:
    def __init__(self):
        self.nets = [] # list of Nets
        self.clusters = [] # list of Clusters
        self.clustersNames = set()
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
        self.clustersNames.add(cluster.name)

    def setWeight(self, index, weight):
        self.weights[index] = weight

    def setWeightNormalized(self, index, weight):
        self.weightsNormalized[index] = weight

    def getTotNetLength(self):
        '''
        Return the cumulated length of all nets contained inside the hyperedge.
        '''
        totLen = 0
        for net in self.nets:
            totLen += net.wl
        return totLen


class Instance:
    def __init__(self, name):
        self.name = name
        self.cluster = None # Cluster object to which this instance belongs
        self.nets = list() # List of Net object that are connecting this instance.
        self.x = 0
        self.y = 0

    def addCluster(self, cluster):
        self.cluster = cluster

    def addNet(self, net):
        self.nets.append(net)

# def initWeightsStr(mydir, metisInputFiles, paritionFiles, partitionDirectivesFiles):
def initWeightsStr(edgeWeightTypesStr, vertexWeightTypesStr):
    '''
    Gives a name to the the different kind of edge weights adn vertex weights.
    '''

    # TODO sub-optimial loop. Why did I use all those if?
    for edgeWeightType in range(0, EDGE_WEIGHTS_TYPES):
        if edgeWeightType == 0:
            edgeWeightTypesStr.append("01_NoWires")
        if edgeWeightType == 1:
            edgeWeightTypesStr.append("02_1-NoWires")
        if edgeWeightType == 2:
            edgeWeightTypesStr.append("03_TotLength")
        if edgeWeightType == 3:
            edgeWeightTypesStr.append("04_1-TotLength")
        if edgeWeightType == 4:
            edgeWeightTypesStr.append("05_AvgLength")
        if edgeWeightType == 5:
            edgeWeightTypesStr.append("06_1-AvgLength")
        if edgeWeightType == 6:
            edgeWeightTypesStr.append("07_NoWiresXTotLength")
        if edgeWeightType == 7:
            edgeWeightTypesStr.append("08_1-NoWiresXTotLength")
        if edgeWeightType == 8:
            edgeWeightTypesStr.append("09_NoWires+TotLength")
        if edgeWeightType == 9:
            edgeWeightTypesStr.append("10_1-NoWires+TotLength")
        if edgeWeightType == 10:
            edgeWeightTypesStr.append("11_fanout")

    if SIMPLE_GRAPH and VERTEX_WEIGHTS_TYPES == 2:
        vertexWeightTypesStr.append("area-power")
    else:
        for vertexWeightType in range(0, VERTEX_WEIGHTS_TYPES):
            if vertexWeightType == 0:
                vertexWeightTypesStr.append("area")
            elif vertexWeightType == 1:
                vertexWeightTypesStr.append("power")



def writeOutput(directory, strToWrite):
    print("Writing into {}".format(directory))
    with open(directory, 'w') as f:
        f.write(strToWrite)


if __name__ == "__main__":
    dirs = []
    EDGE_WEIGHTS_TYPES = 0
    CUSTOM_FIXFILE = False
    UBfactor = 1
    args = docopt(__doc__)
    print(args)
    if args["-d"]:
        dirs.append(args["-d"])
    else:
        # dirs = ["../temp_design/"]
        dirs = ["/home/para/dev/def_parser/2018-01-25_16-16-50/ccx_Naive_Geometric_25"]
    if args["-w"]:
        EDGE_WEIGHTS_TYPES = int(args["-w"])
    else:
       EDGE_WEIGHTS_TYPES = 10
    if args["--seed"]:
        RANDOM_SEED = int(args["--seed"])
    if args["--algo"]:
        ALGO = int(args["--algo"])
    else:
        ALGO = 0
    if args["--path"]:
        if ALGO == 0:
            METIS_PATH = args["--path"]
        elif ALGO == 1:
            PATOH_PATH = args["--path"]
        elif ALGO == 2:
            CIRCUT_PATH = args["--path"]
    if args["--simple-graph"]:
        SIMPLE_GRAPH = True
    if args["--fix-pins"]:
        FIX_PINS = True
    if args["--custom-fixfile"]:
        CUSTOM_FIXFILE = True
    if args["--ub"]:
        UBfactor=args["--ub"]


    # Random seed is preset or random, depends on weither you want the same results or not.
    # Note: even if the seed is set, maybe it won't be enough to reproduce the results since the partitioner may use its own.
    if RANDOM_SEED == 0:
        RANDOM_SEED = random.random()
    random.seed(RANDOM_SEED)
    # print "Seed: " + str(RANDOM_SEED)

    for mydir in dirs:

        output_dir = os.path.join(mydir, "partitions_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(ALGO_DICO[ALGO]))
        try:
            os.makedirs(output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # Load base config from conf file.
        logging.config.fileConfig('log.conf')
        # Load logger from config
        logger = logging.getLogger('default')
        # Create new file handler
        fh = logging.FileHandler(os.path.join(output_dir, 'phoney_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'))
        # Set a format for the file handler
        fh.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
        # Add the handler to the logger
        logger.addHandler(fh)


        logger.info("Output going to %s", output_dir)
        logger.info("Seed: %s", str(RANDOM_SEED))




        # Initialize lists to be used later.
        edgeWeightTypesStr = list()
        vertexWeightTypesStr = list()
        initWeightsStr(edgeWeightTypesStr, vertexWeightTypesStr)

        graph = Graph()
        clusterCount = 500 # Ken's cluster count. Used if CLUSTER_INPUT_TYPE == 1

        if not IMPORT_HYPERGRAPH:
            if CLUSTER_INPUT_TYPE == 0:
                clustersAreaFile = os.path.join(mydir, "ClustersArea.out")
                clustersInstancesFile = os.path.join(mydir, "ClustersInstances.out")
            # elif CLUSTER_INPUT_TYPE == 1:
            #     clustersAreaFile = mydir + "test" + str(clusterCount) + ".area"
            #     clustersInstancesFile = mydir + "test" + str(clusterCount) + ".tcl"
            # elif CLUSTER_INPUT_TYPE == 2:
            #     clustersAreaFile = mydir + "2rpt.clusters.rpt"
            #     clustersInstancesFile = mydir + "ClustersInstances.out"
            else:
                logger.warning("Cluster type not supported.")

            instancesCoordFile = os.path.join(mydir, "CellCoord.out")
            pinCellsFile = os.path.join(mydir, PIN_CELLS_F)
            if args["--netsegments"]:
                netsWL = os.path.join(mydir, "WLnets_segments.out")
                netsInstances = netsWL
            else:
                netsWL = os.path.join(mydir, "WLnets.out")
                netsInstances = os.path.join(mydir, "InstancesPerNet.out")
            pinCoordFile = os.path.join(mydir, PIN_COORD_F)
            memoryBlocksFile = os.path.join(mydir, "bb.out")
            # memoryBlocksFile = mydir + "1.bb.rpt"

            # Extract clusters
            if CLUSTER_INPUT_TYPE == 0:
                # graph.ReadClusters(clustersAreaFile, 14, 2) # Spyglass
                graph.ReadClusters(clustersAreaFile, 1, 0) # def_parser
            elif CLUSTER_INPUT_TYPE == 1:
                graph.ReadClusters(clustersAreaFile, 0, 0)


            # Extract cluster instances
            t0 = time.time()
            graph.readClustersInstances(clustersInstancesFile, 0, 0)
            t1 = time.time()
            logger.debug("time: %s", str(t1-t0))
            graph.instancesCoordinates(instancesCoordFile,0,0)
            if MEMORY_BLOCKS:
                t0 = time.time()
                graph.readMemoryBlocks(memoryBlocksFile, 14, 4)
                t1 = time.time()
                logger.debug("time: %s", str(t1-t0))
            # Begin with the netWL file, as there are less nets there.
            graph.readNetsWireLength(netsWL, 1, 0, segments=args["--netsegments"])
            if args["--netsegments"]:
                graph.readNets(netsInstances, 1, 0, segments=args["--netsegments"])
            else:
                graph.readNets(netsInstances, 0, 0, segments=args["--netsegments"])
            graph.readPins(pinCoordFile)

            t0 = time.time()
            graph.findHyperedges(output_dir)
            t1 = time.time()
            logger.debug("time: %s", str(t1-t0))

            # print "Dumping the graph into " + mydir + DUMP_FILE
            # with open(mydir + DUMP_FILE, 'wb') as f:
            #     pickle.dump(graph, f)

        else:
            logger.info("Loading the graph from %s%s", mydir, DUMP_FILE)
            with open(os.path.join(mydir, DUMP_FILE), 'rb') as f:
                graph = pickle.load(f)

        edgeWeightType = 0
        graph.computeClustersPower()
        graph.computeHyperedgeWeights(True)
        graph.computeVertexWeights()

        partitionFiles = []

        for edgeWeightType in range(0, len(edgeWeightTypesStr)):
            for vertexWeightType in range(0, len(vertexWeightTypesStr)):
                if ALGO == 0: # METIS
                    metisInput = os.path.join(output_dir, "metis_" + edgeWeightTypesStr[edgeWeightType] + \
                        "_" + vertexWeightTypesStr[vertexWeightType] + ".hgr")
                    if CUSTOM_FIXFILE:
                        metisInputFixfile = os.path.join(output_dir, "fixfile.hgr")
                        shutil.copyfile(os.path.join(mydir, "fixfile.hgr"), metisInputFixfile)
                    else:
                        metisInputFixfile = os.path.join(output_dir, "metis_{}_{}_fixfile.hgr".format(edgeWeightTypesStr[edgeWeightType], vertexWeightTypesStr[vertexWeightType]))
                    metisPartitionFile = metisInput + ".part.2"
                    partitionDirectivesFile = metisInput + ".tcl"
                    gatePerDieFile = metisInput + ".part"

                    logger.info("=============================================================")
                    logger.info("> Edge weight: " + edgeWeightTypesStr[edgeWeightType])
                    logger.info("> Vertex weight: " + vertexWeightTypesStr[vertexWeightType])
                    logger.info("=============================================================")

                    graph.generateMetisInput(metisInput, edgeWeightType, vertexWeightType, FIX_PINS, metisInputFixfile, pinCellsFile, CUSTOM_FIXFILE)
                    graph.GraphPartition(metisInput, FIX_PINS, metisInputFixfile, UBfactor)
                    graph.WritePartitionDirectives(metisPartitionFile, partitionDirectivesFile, gatePerDieFile)
                    graph.extractPartitionConnectivity(os.path.join(output_dir, "connectivity_partition.txt"), edgeWeightTypesStr[edgeWeightType])
                    graph.extractPartitionNetLengthCut(os.path.join(output_dir, "cutLength_partition.txt"), edgeWeightTypesStr[edgeWeightType])
                    graph.extractPartitions(partitionDirectivesFile)
                    graph.computePartitionArea()
                    graph.computePartitionPower()
                    if MANUAL_ASYMMETRY:
                        graph.manualOverride()
                    partitionFiles.append(metisPartitionFile)

                elif ALGO == 1: # PaToH
                    patohInput = os.path.join(output_dir, "patoh_" + edgeWeightTypesStr[edgeWeightType] + \
                        "_" + vertexWeightTypesStr[vertexWeightType] + ".hgr")
                    patohPartitionFile = patohInput + ".part.2"
                    partitionDirectivesFile = patohInput + ".tcl"
                    gatePerDieFile = patohInput + ".part"

                    logger.info("=======================PaToH=================================")
                    logger.info("> Edge weight: " + edgeWeightTypesStr[edgeWeightType])
                    logger.info("> Vertex weight: " + vertexWeightTypesStr[vertexWeightType])
                    logger.info("=============================================================")

                    graph.generatePaToHInput(patohInput, edgeWeightType, vertexWeightType)
                    graph.GraphPartitionPaToH(patohInput)
                    graph.WritePartitionDirectives(patohPartitionFile, partitionDirectivesFile, gatePerDieFile)
                    graph.extractPartitionConnectivity(os.path.join(output_dir, "connectivity_partition.txt"), edgeWeightTypesStr[edgeWeightType])
                    graph.extractPartitionNetLengthCut(os.path.join(output_dir, "cutLength_partition.txt"), edgeWeightTypesStr[edgeWeightType])
                    graph.extractPartitions(partitionDirectivesFile)
                    graph.computePartitionArea()
                    graph.computePartitionPower()

                elif ALGO == 2: # Circut
                    circutInput = os.path.join(output_dir, "circut_" + edgeWeightTypesStr[edgeWeightType] + \
                        "_" + vertexWeightTypesStr[vertexWeightType] + ".hgr")
                    circutPartitionFile = circutInput + "_maxcut.cut"
                    partitionDirectivesFile = circutInput + ".tcl"
                    gatePerDieFile = circutInput + ".part"

                    logger.info("=======================Circut=================================")
                    logger.info("> Edge weight: " + edgeWeightTypesStr[edgeWeightType])
                    logger.info("> Vertex weight: " + vertexWeightTypesStr[vertexWeightType])
                    logger.info("==============================================================")

                    graph.generateCircutInput(circutInput, edgeWeightType, vertexWeightType)
                    graph.GraphPartitionCircut(circutInput, circutPartitionFile.split(os.sep)[-1])
                    graph.WritePartitionDirectives(circutPartitionFile, partitionDirectivesFile, gatePerDieFile)
                    graph.extractPartitionConnectivity(os.path.join(output_dir, "connectivity_partition.txt"), edgeWeightTypesStr[edgeWeightType])
                    graph.extractPartitionNetLengthCut(os.path.join(output_dir, "cutLength_partition.txt"), edgeWeightTypesStr[edgeWeightType])
                    graph.extractPartitions(partitionDirectivesFile)
                    graph.computePartitionArea()
                    graph.computePartitionPower()

                elif ALGO == 3: # Custom partitioner
                    custPartInput = os.path.join(output_dir, "custpart_" + edgeWeightTypesStr[edgeWeightType] + \
                        "_" + vertexWeightTypesStr[vertexWeightType] + ".hgr")
                    custPartPartitionFile = custPartInput + ".part.2"
                    partitionDirectivesFile = custPartInput + ".tcl"
                    gatePerDieFile = custPartInput + ".part"

                    logger.info("=======================Custom Part=================================")
                    logger.info("> Edge weight: " + edgeWeightTypesStr[edgeWeightType])
                    logger.info("> Vertex weight: " + vertexWeightTypesStr[vertexWeightType])
                    logger.info("==============================================================")

                    graph.generateCustPartInput(custPartInput, edgeWeightType, vertexWeightType)
                    graph.GraphPartitionCustPart(custPartInput, custPartPartitionFile)
                    graph.WritePartitionDirectives(custPartPartitionFile, partitionDirectivesFile, gatePerDieFile)
                    graph.extractPartitionConnectivity(os.path.join(output_dir, "connectivity_partition.txt"), edgeWeightTypesStr[edgeWeightType])
                    graph.extractPartitionNetLengthCut(os.path.join(output_dir, "cutLength_partition.txt"), edgeWeightTypesStr[edgeWeightType])
                    graph.extractPartitions(partitionDirectivesFile)
                    graph.computePartitionArea()
                    graph.computePartitionPower()

                elif ALGO == 4: # Random partitioner
                    ranPartInput = os.path.join(output_dir, "ranpart_" + edgeWeightTypesStr[edgeWeightType] + \
                        "_" + vertexWeightTypesStr[vertexWeightType] + ".hgr")
                    ranPartPartitionFile = ranPartInput + ".part.2"
                    ranPartPartitionFileStr = ""
                    partitionDirectivesFile = ranPartInput + ".tcl"
                    partitionDirectivesFileStr = ""
                    gatePerDieFile = ranPartInput + ".part"
                    gatePerDieFileStr = ""
                    conPartFile = os.path.join(output_dir, "connectivity_partition.txt")
                    conPartFileStr = ""
                    cutLenPartFile = os.path.join(output_dir, "cutLength_partition.txt")
                    cutLenPartFileStr = ""

                    logger.info("=======================Custom Part=================================")
                    logger.info("> Edge weight: " + edgeWeightTypesStr[edgeWeightType])
                    logger.info("> Vertex weight: " + vertexWeightTypesStr[vertexWeightType])
                    logger.info("==============================================================")

                    # Make dir 'min'
                    minDir = os.path.join(output_dir, "min")
                    try:
                        os.makedirs(minDir)
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise
                    # Make dir 'max'
                    maxDir = os.path.join(output_dir, "max")
                    try:
                        os.makedirs(maxDir)
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise

                    conMin = -1
                    conMax = 0
                    maxIt = 100
                    cnt = 0 # iteration counter
                    i = maxIt

                    fileList = ["connectivity_partition.txt", "cutLength_partition.txt", ranPartPartitionFile.split(os.sep)[-1], partitionDirectivesFile.split(os.sep)[-1], gatePerDieFile.split(os.sep)[-1]]
                    strList = [conPartFileStr, cutLenPartFileStr, ranPartPartitionFileStr, partitionDirectivesFileStr, gatePerDieFileStr] # Needs to be in the same order as fileList.
                    while i > 0:
                        cnt += 1
                        logger.debug("~~~~~~~~~~~~~~~~~~~~~~")
                        logger.debug("~ Random part #{} ~".format(i))
                        logger.debug("~~~~~~~~~~~~~~~~~~~~~~")
                        newCons = list()
                        # Reset strings
                        for j in range(len(strList)):
                            strList[j] = ""

                        strList[2] = graph.GraphPartitionRanPart(ranPartPartitionFile, write_output=False)
                        strList[3], strList[4] = graph.WritePartitionDirectives(ranPartPartitionFile, partitionDirectivesFile, gatePerDieFile, write_output=False, partStr=strList[2])
                        newCon, strList[0] = graph.extractPartitionConnectivity(conPartFile, edgeWeightTypesStr[edgeWeightType], write_output=False)
                        newCons.append(newCon)
                        graph.extractPartitions(partitionDirectivesFile, readFile=False, partStr=strList[3])
                        if newCon < conMin or conMin == -1:
                            conMin = newCon
                            strList[1] = graph.extractPartitionNetLengthCut(cutLenPartFile, edgeWeightTypesStr[edgeWeightType], write_output=False)
                            graph.computePartitionArea()
                            graph.computePartitionPower()
                            # Write all outputs to 'min'
                            for j in range(len(fileList)):
                                writeOutput(os.path.join(minDir, fileList[j]), strList[j])
                            i = maxIt
                        # if newCon > conMax:
                        #     conMax = newCon
                        #     strList[1] = graph.extractPartitionNetLengthCut(cutLenPartFile, edgeWeightTypesStr[edgeWeightType], write_output=False)
                        #     graph.computePartitionArea()
                        #     graph.computePartitionPower()
                        #     # Writes all outputs to 'max'
                        #     for j in range(len(fileList)):
                        #         writeOutput(os.path.join(maxDir, fileList[j]), strList[j])
                        #     # i = maxIt
                        i -= 1
                    logger.info("Final mincut: {}, final maxcut: {}, average: {}, iterations: {}".format(conMin, conMax, statistics.mean(newCons), cnt))
                    # with open(os.path.join(output_dir, "connectivities.out"), 'w') as f:
                    #     strCon = ""
                    #     for newCon in newCons:
                    #         strCon += "{}\n".format(newCon)
                    #     f.write(strCon)



        # graph.hammingReport(partitionFiles)
        # graph.plotWeights()
