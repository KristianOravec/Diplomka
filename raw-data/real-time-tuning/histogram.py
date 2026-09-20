#!/usr/bin/env python

"""Histogram generator

Usage:
  histogram.py -s <source> -t <time>

Options:
  -h Show this screen.
  -s Source logs directory.
  -t Time in seconds used in histogram.

"""

from docopt import docopt
from os import listdir
import numpy

if __name__ == '__main__':
    arguments = docopt(__doc__)
    #f = open(arguments['-s'], "r")
    files = listdir(arguments['-s'])
    maxTime = int(arguments['-t'])
    times = [[] for _ in range(maxTime)]
    confs = [[] for _ in range(0, 1000000)]
    maxConf = 0

    for fl in files :
        f = open(arguments['-s'] + "/" + fl, "r")
        perfT = [0]*(maxTime+1)
        perfC = []
        lastIdx = 0
        for line in f.readlines():
            words = line.split(' ')
            if words[0] == "Execution":
                idx = int(words[2])
                if (idx <= maxTime) :
                    perfT[idx] = int(words[9])
                    for i in range(lastIdx+1, idx) :
                        perfT[i] = perfT[lastIdx]
                    lastIdx = idx
                    conf = int(words[5])
                    if conf >= len(perfC) :
                        perfC.append(int(words[9]))
                    else :
                        perfC[len(perfC)-1] = int(words[9]);
        f.close()
        for i in range(0, maxTime) :
            if perfT[i] > 0 :
                times[i].append(perfT[i])
        maxConf = max(conf, len(perfC))
        for i in range(0, len(perfC)) :
            confs[i].append(perfC[i])

    # for practical reason, ommit 1st second (typically no data)
    print("time(s) mean stddev min max")
    for i in range(1, maxTime) :
        print(str(i) + " " + str(numpy.mean(times[i])) + " " + str(numpy.std(times[i])) + " " + str(numpy.min(times[i])) + " " + str(numpy.max(times[i])))

    print("conf mean stddev min max")
    for i in range(1, maxConf) :
        print(str(i) + " " + str(numpy.mean(confs[i])) + " " + str(numpy.std(confs[i])) + " " + str(numpy.min(confs[i])) + " " + str(numpy.max(confs[i])))

