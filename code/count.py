#!/usr/bin/python
from __future__ import with_statement

import datetime
import os
import os.path
import re
import sys
import time

# INPUTS
CANON_NOTES_PATH = "../data/nursenotes26_canonicalized"
NUM_FILES_LIMIT = None
FEATURES_PATH = "../data/dPatientsExport.csv"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S EST"

# OUTPUTS
OUT_PATH = "../out/"
VOCAB_PATH = os.path.join(OUT_PATH, "vocabulary.txt")
FEAT_PATH = os.path.join(OUT_PATH, "feature.txt")
MATRIX_PATH = os.path.join(OUT_PATH, "patient_data.txt")
MATRIXT_PATH = os.path.join(OUT_PATH, "patient_data_temporal.txt")
MATRIXR_PATH = os.path.join(OUT_PATH, "patient_rows.txt")

def main():
    # Print the variables being used for inputs
    print "Using %s canonicalized notes from %s" % (NUM_FILES_LIMIT or "ALL", CANON_NOTES_PATH)
    print "Using outcomes from %s" % (FEATURES_PATH)
    print
    starttime = time.time()

    # Get the list of filepaths (qualified)
    filepaths = filter(os.path.isfile,
        (os.path.join(CANON_NOTES_PATH, x) for x in os.listdir(CANON_NOTES_PATH))
        )
    filepaths = filepaths[:NUM_FILES_LIMIT]

    # Get the heads and features from doc
    with open(FEATURES_PATH) as f:
        fheads = dict(
            (w.replace('"',''), i) for i,w in enumerate(f.readline().strip().split(","))
            )
        features = dict(
            (cline[0], cline) for cline in
                [line.strip().split(",") for line in f.readlines()]
            )

    # Parse each patient record
    print "Extracting word counts..."
    vocab = {}  # Map word -> col_idx
    vocab_idx = 1   # Current col_idx
    row_num = 1 # Current row number (i.e. patient)
    row_add = 0 # Current row adds (i.e. when patients temporally split)
    for i, doc in enumerate(sorted(filepaths)):
        if i % 100 == 0:
            print "%d..\r" % (i),
            sys.stdout.flush()

        # Correlate files to the appropriate features
        sid = re.search(r"-0*(\d+).txt", doc)
        sid = sid.group(1)
        if sid not in features: 
            continue
        with open(FEAT_PATH, 'a') as f:
            print >>f, "\t".join(features[sid])

        # Read heads and notes from doc
        with open(doc) as f:
            # Read in the headers
            nheads = f.readline().strip().split("_:-:_")
            heads = nheads[:-1]

            # Create a regex to match notes from the heads
            regex = "^" + "_:-:_".join("(?P<%s>.*?)" % (x) for x in heads) + "$"
               
            # Read the notes
            matches = re.finditer(regex, f.read(), re.MULTILINE | re.DOTALL)
            notes = [m.groupdict() for m in matches]

        # Calculate wordcounts
        gwc = {}    # global
        min_date = min(
            [datetime.datetime.strptime(n['CHARTTIME'], DATE_FORMAT) for n in notes]
            )
        notes = sorted(notes, 
            key=lambda n: datetime.datetime.strptime(n['CHARTTIME'], DATE_FORMAT)
            )
        for note in notes:
            lwc = {}    # local
            for line in note['TEXT'].split('\n'):
                word = line.strip()
                # Add word to vocabulary if necessary
                if word not in vocab:
                    vocab[word] = vocab_idx
                    vocab_idx += 1
                    with open(VOCAB_PATH, 'a') as f:
                        print >>f, word

                #Increment word count
                lwc[word] = lwc.get(word, 0) + 1

            # Output local wordcount
            for w, c in sorted(lwc.iteritems(), key=lambda x: vocab[x[0]]):
                with open(MATRIXT_PATH, 'a') as f:
                    print >>f, "%d\t%d\t%d" % (row_num + row_add, vocab[w], c)
                gwc[w] = gwc.get(w, 0) + c
            row_add += 1

            # Calculate hours since first note
            td = datetime.datetime.strptime(note['CHARTTIME'], DATE_FORMAT) - min_date
            secs = (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6
            hrs = secs / 3600
            
            with open(MATRIXR_PATH, 'a') as f:
                print >>f, "%s\t%d" % (
                    "\t".join((note[h] for h in heads if h != 'TEXT')), hrs
                    )

        # Output global wordcount
        for w,c in sorted(gwc.iteritems(), key=lambda x: vocab[x[0]]):
            with open(MATRIX_PATH, 'a') as f:
                print >>f, "%d\t%d\t%d" % (row_num, vocab[w], c)
        row_num += 1
   
    # Print analysis
    stoptime = time.time()
    print "Done analyzing %d documents in %.2f seconds (%.2f docs/sec)" % (i, stoptime - starttime, i/(stoptime-starttime))
    print "Used %d/%d documents" % (row_num, i)


if __name__ == "__main__":
    main()
