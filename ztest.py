import json

locs = []
visited = set()

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def search(correct, name):
    for c in correct:
        if c['location'] == name:
            return c

    return {}


correct = load_json("correct.json")
data = load_json("/home/ubuntu/si_media/export/test_biz/new_result.json")
for d in data.values():
    for k in ["place_of_receipt", "place_of_delivery", "port_of_loading", "port_of_discharge", "final_destination"]:
        if d[k] and d[k] not in visited:
            item = search(correct, d[k])
            locs.append({
                "location": d[k],
                "expect": item.get("expect"),
            })
            visited.add(d[k])

with open("test_location.json", "w") as f:
    json.dump(locs, f)
