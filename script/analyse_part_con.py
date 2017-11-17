print "######################"
print "The aim of this script is to read te file 'connectivity_partition.txt' \
and find out if the nets cut by the partition are different from one metric \
to the other. \
Then, it extracts the nets info from a WLnets.out file."
print "######################"

fileLines = [] #List of all the lines in the file
partitions = [] #List of sets
with open("connectivity_partition.txt", 'r') as f:
	fileLines = f.readlines()

for line in fileLines:
	if line != "\n":
		partitions.append(set(line.strip().split(',')[2:]))
print len(partitions[0].difference(partitions[3]))
print len(partitions[3].difference(partitions[0]))
print len(partitions[0].symmetric_difference(partitions[3]))
print len(partitions[0].intersection(partitions[3]))

merge = partitions[0].intersection(partitions[1])

for i in range(2,len(partitions)):
	merge = merge.intersection(partitions[i])

print "Total common nets across all metrics: " + str(len(merge))
# print merge


# Extract the net info from WLnets.out.
wlnetsfile = "/home/para/dev/metis_unicorn/temp_design/WLnets.out"
wlnetsinfo = dict() # ('Name', [pins, length])

with open(wlnetsfile, 'r') as f:
	fileLines = f.readlines()

for i in xrange(1, len(fileLines)):
	# Skip the first line, it's just the column headers.
	# The first element of the line if the key, the rest is added as an array
	wlnetsinfo[fileLines[i].strip().split()[0]] = fileLines[i].strip().split()[1:]

# print wlnetsinfo

# Get the net info about the common nets.
for net in merge:
	print net + ", pins: " + str(wlnetsinfo[net][0]) + ", length: " + str(wlnetsinfo[net][1])