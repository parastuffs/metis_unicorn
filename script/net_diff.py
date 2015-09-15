INPUT_FILE_DIR = "../input_files/"
forbiddenLines = ["-", "#", "NAME", "NET", "Steiner"]


nets = list()
WLnets = list()
missingNets = list()

with open(INPUT_FILE_DIR+'Nets.out', 'r') as netsInput:
	nets = netsInput.read().splitlines()

with open(INPUT_FILE_DIR+'WLnets.out', 'r') as WLnetsInput:
	for line in WLnetsInput:
		forbidden = False
		for forbiddenChar in forbiddenLines:
			if line.startswith(forbiddenChar):
				forbidden = True
		if not forbidden:
			line = line.strip(' \n')
			lineSplitted = line.split(' ')
			WLnets.append(lineSplitted[0])

for net in nets:
	try:
		WLnets.index(net)
	except:
		missingNets.append(net)

# print(WLnets)
print(missingNets)

print len(missingNets),"missing nets in WLnets.out out of", len(nets)