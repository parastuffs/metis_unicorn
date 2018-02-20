import sys, getopt, os

print "######################"
print "The aim of this script is to read te file 'cutLength_partition.txt' \
and print the total length cut into a .csv"
print "######################"

infile = ""
try:
    opts, args = getopt.getopt(sys.argv[1:],"f:")
except getopt.GetoptError:
    print "YO FAIL"
else:
    for opt, arg in opts:
        if opt == "-f":
            infile = arg

    if infile == "":
        infile = "cutLength_partition.txt"

fileLines = [] #List of all the lines in the file
csvout = ""
i = 0
with open("cutLength_partition.txt", 'r') as f:
	fileLines = f.readlines()

'''
Input format of the cutLength_partition.txt file:
<number> clusters, <number> graphTotLen,<edge weight type> <total length cut>

e.g.: 4 clusters, 97616 graphTotLen,01_NoWires 60720

Output format:
<clusters>, <graph tot len>, <tot len 1>, <...>, <tot len 10>
'''

for line in fileLines:
	if line != "\n":
		if i == 0:
			csvout += str(line.strip().split(' ')[0]) + ","
			csvout += str(line.strip().split(' ')[2]) + ","
		csvout += str(line.strip().split(' ')[5]) + ','
		i += 1
		if i == 10: # 10 is the number of different weight types.
			# End of the clustering, new line.
			i = 0
			csvout += "\n"

with open(os.path.join("/".join(infile.split('/')[:-1]),"cutLength_partition_wl.csv") ,'w') as f:
	f.write(csvout)