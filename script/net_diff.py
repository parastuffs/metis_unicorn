""" net_diff.py
2015-09-15

Compares the network list in Nets.out and WLnets.out.
Lists all the nets that are in Nets.out but not in WLnets.out.
"""

INPUT_FILE_DIR = "../input_files/"
FORBIDDEN_LINES = ["-", "#", "NAME", "NET", "Steiner"] # Strings beginning unwanted lines


nets = list()
WLnets = list()
missingNets = list()

# Get the nets
with open(INPUT_FILE_DIR+'Nets.out', 'r') as netsInput:
	nets = netsInput.read().splitlines()

# Get the WLnets...
with open(INPUT_FILE_DIR+'WLnets.out', 'r') as WLnetsInput:
	for line in WLnetsInput:
		#... but only if they do not begin with a forbidden char.
		forbidden = False
		for forbiddenChar in FORBIDDEN_LINES:
			if line.startswith(forbiddenChar):
				forbidden = True
		if not forbidden:
			line = line.strip(' \n')
			lineSplitted = line.split(' ')
			# Keep only the first column
			WLnets.append(lineSplitted[0])

# If the net from the list nets is not found in the list WLnets,
# add it to the list missingNets.
for net in nets:
	try:
		WLnets.index(net)
	except:
		missingNets.append(net)

print(missingNets)

print len(missingNets),"missing nets in WLnets.out out of", len(nets)