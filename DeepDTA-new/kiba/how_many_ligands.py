import json

# Open and read the text file
with open('C:/Users/admin/Desktop/DeepDTA-master/data/kiba/ligands_iso.txt', 'r') as file:
    data = file.read()

# Parse the data as JSON
try:
    data_dict = json.loads(data)
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    data_dict = {}

# Count the entries with keys starting with "CHEMBL"
chembl_count = sum(1 for key in data_dict.keys() if key.startswith("CHEMBL"))

print(f"Number of CHEMBL entries: {chembl_count}")
