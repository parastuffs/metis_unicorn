print "######################"
print "The aim of this script is to read te file 'connectivity_partition.txt' \
and print the number of net cut into a .csv"
print "######################"

fileLines = [] #List of all the lines in the file
csvout = ""
i = 0
with open("connectivity_partition.txt", 'r') as f:
	fileLines = f.readlines()

for line in fileLines:
	if line != "\n":
		i += 1
		csvout += str(line.strip().split(' ')[1]) + ","
		if i == 10:
			# End of the clustering, new line.
			i = 0
			csvout += "\n"

with open("connectivity_partition_wl.csv" ,'w') as f:
	f.write(csvout)