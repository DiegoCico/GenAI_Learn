# data_loader.py
import json

def load_data(path="baby_bussiness/business_data.json"):
    with open(path, "r") as f:
        return json.load(f)
