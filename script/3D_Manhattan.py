"""
3D Manhattan distance approximation.
This script extracts the gates and partitions information from files generated by def_parser and phoney.
The first step is to determine the 2D manhattant distance between every pair of separated gates.

Usage:
    3D_Manhattan.py     [-g <CellCoord.out file>] [-p <Partition file>] [-c <ClustersInstances.out file>] [-n <connectivity_partition.txt file>]
    3D_Manhattan.py     --help

Options:
    -g <file>   Gates coordinates  
    -p <file>   Partitions directives
    -c <file>   Which gates inside which cluster
    -n <file>   List of cut nets
    -h --help   Print this help

Details:
The cell coordinates file should be as follows:
net_name,gate_name,gate_x,gate_y

The partitions directive file should be as follows:
add_to_die -cluster <cluster id> -die Die<O,1>

The cluster instances file should be as follows:
<cluster id> <gate_name 1> ... <gate_name n>

The connectivity partition file should be as follow:
<Amount of clusters> clusters, <Total number of nets> graphTotNets, <weight i> <number of n nets cut>, <net cut 1>, ... <net cut n>

"""
from __future__ import division
from docopt import docopt
import re
import statistics
import math
import numpy as np
import os
import matplotlib.pyplot as plt

# TODO this analysis is only done for the first partitioning weight (first line in connectivity_partition.txt file).

def Evaluate3DLength(partFile, clusterFile, cellFile, netCutFile):
    partClust = dict() # key: cluster id, value: partition id
    partGates = dict() # key: gate name, value: partition id
    gatesCoord = dict() # key: gate name, value: tuple (x,y)
    NetsGates = dict() # key: net name, value: array of gate names
    NetsCut = list() # List of nets cut, out of netCutFile
    manDists = list() # List of all manhattan distances

    # Retrieve in which partition are placed each clusters
    with open(partFile, 'r') as f:
        lines = f.readlines()
    p = re.compile("Die([0-9]+)")
    for line in lines:
        partClust[int(line.split()[2])] = int(p.search(line.split()[4]).group(1))

    # Retrieve in which partition are placed each gates
    with open(clusterFile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        gates = line.split()
        clusterID = int(gates[0])
        for i in range(1,len(gates)):
            partGates[gates[i]] = partClust[clusterID]

    # Retrieve the coordinates of each gates
    with open(cellFile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        gatesCoord[line.split(',')[1]] = (float(line.split(',')[2]), float(line.split(',')[3]))


    # Retrieve the net
    with open(cellFile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        a = list()
        netName = line.split(',')[0]
        if netName in NetsGates:
            for n in NetsGates[netName]:
                a.append(n)
        a.append(line.split(',')[1])
        NetsGates[netName] = a[:]

    with open(netCutFile, 'r') as f:
        lines = f.readlines()
    NetsCut = lines[0].strip().split(',')[3:]

    for net in NetsCut:
        gates = NetsGates[net]
        # There should always be at least two gates in a cut net.
        for i in range(len(gates)):
            for j in range(i, len(gates)):
                # If the two gates belong to different partitions
                if partGates[gates[i]] != partGates[gates[j]]:
                    # Manhattan distance is the half-perimeter length of the bounding box containing the two gates.
                    # In other words, the sum of the difference between their two coordinates.
                    manDist = abs(gatesCoord[gates[i]][0] - gatesCoord[gates[j]][0]) + abs(gatesCoord[gates[i]][1] - gatesCoord[gates[j]][1])
                    manDists.append(manDist)


    # Export to csv
    manDistsFile = os.sep.join(netCutFile.split(os.sep)[:-1] + ["netsCutManhattan2D.csv"])
    manDistsStr = "2D Manhattan distance of partitioned gates\n"
    for i in manDists:
        manDistsStr += str(i) + "\n"
    with open(manDistsFile, 'w') as f:
        f.write(manDistsStr)



    # Average
    avgMan = statistics.mean(manDists)
    width = avgMan * (1.0 - (1.0/math.sqrt(2)))
    print("With an average 2D Manhattan distance of {} um, the 3D width should be at most {} um.".format(avgMan, width))

    # First quartile
    q1Man = np.percentile(manDists, 25)
    width = q1Man * (1.0 - (1.0/math.sqrt(2)))
    print("With 1st quartile 2D Manhattan distance of {} um, the 3D width should be at most {} um.".format(q1Man, width))

    # Third quartile
    q3man = np.percentile(manDists, 75)
    width = q3man * (1.0 - (1.0/math.sqrt(2)))
    print("With 1st quartile 2D Manhattan distance of {} um, the 3D width should be at most {} um.".format(q3man, width))


    # Boxplot des distances de Manhattan des fils coupes.
    # plt.boxplot(manDists)
    # plt.show()

    # Plot 3D width as a function of Manhattan distance
    xValues = np.linspace(min(manDists),max(manDists),50)
    yValues = list()
    for x in xValues:
        yValues.append(x * (1 - (1/math.sqrt(2))))
    plt.plot(xValues, yValues)
    plt.show()

    # Plot 3D width as a function of the number of wires cut
    min3DWidth = min(manDists) * (1 - (1/math.sqrt(2)))
    max3DWidth = max(manDists) * (1 - (1/math.sqrt(2)))
    yValues = np.linspace(min3DWidth,max3DWidth,50)
    xValues = list()
    manDists.sort()
    for y in yValues:
        xValues.append(len([i for i in manDists if i > (y / (1 - 1/math.sqrt(2)))]))
    print(xValues)
    print(yValues)
    plt.plot(xValues, yValues)
    # plt.show()









if __name__ == "__main__":
    
    args = docopt(__doc__)
    # print args
    cellFile = "/home/para/dev/def_parser/2019-01-23_22-12-16_ldpc_hierarchical-geometric/ldpc_hierarchical-geometric_32/CellCoord.out"
    partFile = "/home/para/dev/def_parser/2019-01-23_22-12-16_ldpc_hierarchical-geometric/ldpc_hierarchical-geometric_32/partitions_2019-06-20_16-51-03_hMetis/metis_01_NoWires_area.hgr.tcl"
    clusterFile = "/home/para/dev/def_parser/2019-01-23_22-12-16_ldpc_hierarchical-geometric/ldpc_hierarchical-geometric_32/ClustersInstances.out"
    netCutFile = "/home/para/dev/def_parser/2019-01-23_22-12-16_ldpc_hierarchical-geometric/ldpc_hierarchical-geometric_32/partitions_2019-06-20_16-51-03_hMetis/connectivity_partition.txt"
    if args["-g"]:
        cellFile = args["-g"]
    if args["-p"]:
        partFile = args["-p"]
    if args["-c"]:
        clusterFile = args["-c"]
    if args["-n"]:
        netCutFile = args["-n"]

    Evaluate3DLength(partFile, clusterFile, cellFile, netCutFile)






