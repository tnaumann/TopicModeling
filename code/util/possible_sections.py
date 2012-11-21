#!/usr/bin/python

from __future__ import with_statement
import os
import os.path
import re
import string
import sys
import time

# INPUTS
NURSE_NOTES_PATH = 'nursenotes26_filter1'
NUM_FILES_LIMIT = None
REGEX = '(?P<space>\s*)(?P<section>[A-Z][^0-9#*,.;]*?):'

# OUTPUTS
OUT_PATH = 'output.csv'

class counter():
    def __init__(self, _total=0, _blank=0, _start=0):
        self.total = _total
        self.blank = _blank
        self.start = _start

def main():
    # Print the variables being used
    print "Using %s nursing notes from %s" % (NUM_FILES_LIMIT or "ALL", NURSE_NOTES_PATH)
    print
    starttime = time.time()

    filenames = filter(lambda x: os.path.isfile(os.path.join(NURSE_NOTES_PATH, x)),
        os.listdir(NURSE_NOTES_PATH)
        )
    filenames = filenames[:NUM_FILES_LIMIT]

    # Parse each patient record
    sections = {}
    print "Finding sections..."
    for i, e in enumerate(filenames):
        if i % 100 == 0:
            print "%d..\r" % (i),
            sys.stdout.flush()

        with open(os.path.join(NURSE_NOTES_PATH, e)) as f:
            blank = False
            for line in f:
                if re.match(r'^\s*$', line) or re.search(r'_:-:_', line):
                    blank = True
                    continue

                for m in map(lambda x: x.groupdict(), re.finditer(REGEX, line)):
                    section = string.lower(m['section'])
                    if section not in sections:
                        sections[section] = counter()
                    sections[section].total += 1
                    if blank: 
                        sections[section].blank += 1
                    if not m['space']:
                        sections[section].start += 1
                blank = False

    # Write analysis
    with open(OUT_PATH, 'w') as f:
        print >>f, "Section,Total,Start,Blank"
        for k,v in sections.iteritems():
            print >>f, "\"%s\",%d,%d,%d" % (k, v.total, v.start, v.blank)

    # Print analysis
    stoptime = time.time()
    print
    print "Done analyzing %d documents in %.2f seconds (%.2f docs/sec)" % (i, stoptime - starttime, i / (stoptime - starttime))

if __name__ == "__main__":
    main()
