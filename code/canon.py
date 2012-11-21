#!/usr/bin/python
from __future__ import with_statement

import nltk
import os
import os.path
import re
import string
import sys
import time

# INPUTS
NURSE_NOTES_PATH = "../data/nursenotes26"
NUM_FILES_LIMIT = 100
STOPWORDS_LIST_PATH = "../data/onix_stopwords_list.txt"

# OUTPUTS
CANON_NOTES_PATH = NURSE_NOTES_PATH + "_canonicalized"

# Canonicalizes text
def canonicalized(text, stopwords):
    #text = nltk.sent_tokenize(text) # nltk sentences
    #text = map(nltk.word_tokenize, text)    # nltk words
    text = text.split('\n')
    text = map(lambda x: x.split(), text)
    text = [word for sent in text for word in sent] # flatten list
    text = [s.translate(string.maketrans("",""), string.punctuation) for s in text] # rm punct
    text = [s.translate(string.maketrans("",""), string.digits) for s in text] # remove numbers
    text = filter(lambda x: x, text)    # rm nulls
    text = filter(lambda x: re.search("[A-Za-z]+", x), text)    # keep alphas
    text = map(str.lower, text) # lowercase
    text = filter(lambda x: x not in stopwords, text)   # rm stopwords
    text = "\n".join(text)
    return text

def main():
    # Print the variables being used for inputs
    print "Using %s nursing notes from %s" % (NUM_FILES_LIMIT or "ALL", NURSE_NOTES_PATH)
    print "Using stopwords from %s" % (STOPWORDS_LIST_PATH)
    print 
    starttime = time.time()

    # Get the stopwords
    with open(STOPWORDS_LIST_PATH) as f:
        stopwords = set(word.strip() for word in f)

    # Get the list of filenames (unqualified)
    filenames = filter(lambda x: os.path.isfile(os.path.join(NURSE_NOTES_PATH, x)), 
        os.listdir(NURSE_NOTES_PATH)
        )
    filenames = filenames[:NUM_FILES_LIMIT]

    # Parse each patient record
    print "Canonicalizing documents..."
    for i, doc in enumerate(filenames):
        if i % 100 == 0:
            print "%d..\r" % (i),
            sys.stdout.flush()

        # Read heads and notes from doc
        with open(os.path.join(NURSE_NOTES_PATH, doc)) as f:
            # Read in the headers
            nheads = f.readline().strip().split("_:-:_")
            heads = nheads[:-1]

            # Create a regex to match notes from the headers
            regex = "^" + "_:-:_".join("(?P<%s>.*?)" % (x) for x in heads) + "$"

            # Read the notes
            matches = re.finditer(regex, f.read(), re.MULTILINE | re.DOTALL)
            notes = [m.groupdict() for m in matches]

        # Canonicalize
        for note in notes:
            note['TEXT'] = canonicalized(note['TEXT'], stopwords)

        # Write heads and notes to new doc
        with open(os.path.join(CANON_NOTES_PATH, doc), 'w') as f:
            # Print the headers
            print >>f, "_:-:_".join(nheads)

            # Print the notes
            for note in notes:
                print >>f, "_:-:_".join(note[h] for h in heads)

    # Print analysis
    stoptime = time.time()
    print "Done analyzing %d documents in %.2f seconds (%.2f docs/sec)" % (i, stoptime - starttime, i / (stoptime - starttime))
    print "Canonicalized documents in %s" % (CANON_NOTES_PATH)
            
if __name__ == "__main__":
    main()
