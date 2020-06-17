# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:20:27 2020

@author: Radu
"""

import urllib.request
import json  

netid = "radufm2"
alias = "radufm2"
file_path = "radufm2_results.txt"

with open(file_path, 'r') as inputfile:
    alias = inputfile.readline().strip()
    labels = [lbl.strip() for lbl in inputfile]
    
body = {
    'netid': netid,
    'alias': alias,
    'results': [{
        'error': None,
        'dataset': 'hygiene',
        'results': labels
    }]
}

myurl = "http://capstone-leaderboard.centralus.cloudapp.azure.com/api"
req = urllib.request.Request(myurl)
req.add_header('Content-Type', 'application/json; charset=utf-8')
jsondata = json.dumps(body)
jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
req.add_header('Content-Length', len(jsondataasbytes))
print (jsondataasbytes)
response = urllib.request.urlopen(req, jsondataasbytes)

# 35	radufm2	0.6028	0.566	0.6148	0.5646	0.5849	0.568	2020-04-19 | 19:38:33	2

