import re

def readnotes(path):
    """
    Reads headers and notes from a NOTE-EVENTS-*.txt file
    """
    with open(path) as f:
        nheads = f.readline().strip().split("_:-:_")
        heads = nheads[:-1]
        regex = "^" + "_:-:_".join("(?P<%s>.*?)" % (h) for h in heads) + "$"
        matches = re.finditer(regex, f.read(), re.MULTILINE | re.DOTALL)
    notes = map(lambda m: m.groupdict(), matches)
    return nheads, notes


def writenotes(path, nheads, notes, heads=None):
    """
    Write to a path note headers and notes
    """
	if not heads:
		heads = nheads[:-1]
	with open(path, 'w') as f:
		print >>f, "_:-:_".join(nheads)
		for note in notes:
			print >>f, "_:-:_".join(note[h] for h in heads)




	
