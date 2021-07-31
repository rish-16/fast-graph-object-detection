import json, os

MASTER_DIR = '../dataset/MASTERLIST.json'

with open(MASTER_DIR, 'r') as f:
    master_data = json.load(f)