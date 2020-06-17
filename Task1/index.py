import math
import sys
import time

import metapy
import pytoml

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    print('Building or loading index...')
    start_time = time.time()
    idx = metapy.index.make_inverted_index(cfg)
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
