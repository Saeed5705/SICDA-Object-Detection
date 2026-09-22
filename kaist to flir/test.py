import json

path=r"E:\saeedwork\Kaist\kaist_annotations.JSON"

with open(path,"r") as f:
    data=json.load(f)

print(data["categories"])