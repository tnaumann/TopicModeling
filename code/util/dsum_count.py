#!/usr/bin/python
from __future__ import with_statement

import os
import os.path
import re
import string
import sys
import time

# INPUTS
NURSE_NOTES_PATH = "/scratch/mghassem/nursenotes26"
NUM_FILES_LIMIT = None

def main(search_path = NURSE_NOTES_PATH):
    # Print the variables being used for inputs
    print "Using %s nursing notes from %s" % (NUM_FILES_LIMIT or "ALL", NURSE_NOTES_PATH)
    print 
    starttime = time.time()

    # Get the list of filenames (unqualified)
    filenames = filter(lambda x: os.path.isfile(os.path.join(search_path, x)), 
        os.listdir(search_path)
        )
    filenames = filenames[:NUM_FILES_LIMIT]

    # Parse each patient record
    dsum_count = {}
    print "Searching documents..."
    for i, doc in enumerate(filenames):
        if i % 100 == 0:
            print "%d..\r" % (i),
            sys.stdout.flush()

        # Read heads and notes from doc
        with open(os.path.join(search_path, doc)) as f:
            # Read in the headers
            nheads = f.readline().strip().split("_:-:_")
            heads = nheads[:-1]

            # Create a regex to match notes from the headers
            regex = "^" + "_:-:_".join("(?P<%s>.*?)" % (x) for x in heads) + "$"

            # Read the notes
            matches = re.finditer(regex, f.read(), re.MULTILINE | re.DOTALL)
            notes = [m.groupdict() for m in matches]

        dsums = filter(lambda n: n['CATEGORY'] == 'DISCHARGE_SUMMARY', notes)
        dsum_count[len(dsums)] = dsum_count.get(len(dsums), []) + [doc]


    # Print analysis
    stoptime = time.time()
    print
    print "Done analyzing %d documents in %.2f seconds (%.2f docs/sec)" % (i, stoptime - starttime, i / (stoptime - starttime))
    
    return dsum_count
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dsum_count = main(sys.argv[1])
    else:
        dsum_count = main()
    for k,v in dsum_count.iteritems():
        print k, len(v), v[:5]
