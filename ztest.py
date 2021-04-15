import re 

rex = re.compile('(CODE(S|\(S\))?|NO|SYSTEM)', re.IGNORECASE|re.VERBOSE)

found = rex.findall(
    "CONTAINER SAID TO CONTAIN\n1000 CASES\nSOAP\nHS-Codes:3401200000\nAES: X20201210320761\nHS-Code(s):3401200000\nALL MENTIONED CONTAINERS\nSHIPPER'S LOAD, COUNT AND\nSEA".upper()
)

print(found)
print(found[0])