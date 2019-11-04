# metis_unicorn
Implementation of hMETIS

## Weight types

* 0: 01_NoWires_Area
* 1: 02_1_NoWires_Area
* 2: 03_TotLength_Area
* 3: 04_1_TotLength_Area
* 4: 05_AvgLength_Area
* 5: 06_1_AvgLength_Area
* 6: 07_NoWiresXTotLength_Area
* 7: 08_1_NoWiresXTotLength_Area
* 8: 09_NoWires+TotLength_Area
* 9: 10_1_NoWires+TotLength_Area

## Input files

### ClustersArea.out
Header: ```Name Type InstCount Boundary Area```

- Column 1: ```<cluster name>```, same as in ```Clusters.out```
- Column 2: Type of cluster, ```exclusive``` by default.
- Column 3: Amount of instances in the cluster [integer]
- Column 4: Coordinates of the cluster's bottom left corner, ```(<x coordinate
	[float]>,<y coordinate [float]>)```
- Column 5: Coordinates of the cluster's top right corner, ```(<x coordinate
	[float]>,<y coordinate [float]>)```
Columns 4 and 5 only make sense if the cluster is rectangular shaped.
- Column 6: area of the cluster in µm2 [float]

### ClustersInstances.out
Each line: ```<cluster name> <instance name 1> <...> <instance name n>```.

### InstancesPerNet.out
Each line is ```<net name> <instance name 1> <...> <instance name n>```.

### WLnets.out
Header: ```NET NUM_PINS LENGTH```

Then each line is ```<net name> <number of pins [integer]> <length in µm [float]>```.

### bb.out
Memory block file specific to some designs.
The relevant lines should be as follows:
```<instance name> <instance type> <amount of such instance> <total area of the gates in the instance> <total physical area of the instance> <porosity [default: unspecified]> <total area of the cumulated instance> <width> <height> <orientation> <type of standard cells inside>```

## Output files

### .part.2

### .cut

### .tcl

### .part

### .hgr

### connectivity_partition.txt

### cutLength_partition.txt

### raw_hyperedges.out