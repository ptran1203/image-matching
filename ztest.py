import re 

x = "HS21390 FRESH POMEGRANATES\n\nTOTAL: 2720 CTNS\n\nHSCODE081090"

for num in re.findall(r"[a-zA-Z]{1}[0-9]{1,}",x):
    num = num[1:]
    x = x.replace(num, f" {num}")

print(x)