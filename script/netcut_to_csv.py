import sys, getopt, os

print "######################"
print "The aim of this script is to read te file 'connectivity_partition.txt' \
and print the number of net cut into a .csv"
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
        infile = "connectivity_partition.txt"



fileLines = [] #List of all the lines in the file
csvout = ""
i = 0
with open(infile, 'r') as f:
	fileLines = f.readlines()

for line in fileLines:
	if line != "\n":
		if i == 0:
			csvout += str(line.strip().split(' ')[0]) + ","
			csvout += str(line.strip().split(' ')[2]) + ","
		i += 1
		csvout += str(line.strip().split(' ')[5]) + ","
		if i == 10:
			# End of the clustering, new line.
			i = 0
			csvout += "\n"

with open(os.path.join("/".join(infile.split('/')[:-1]),"connectivity_partition_wl.csv") ,'w') as f:
	f.write(csvout)