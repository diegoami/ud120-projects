import csv
import sys
sys.path.append("../tools/")

from work_data_proc import *
from work_parameters import *

def export_to_csv(data_dict, filename, features):
    with open(filename, "wb") as csvfile:
        cwriter = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        cwriter.writerow(['Name','poi']+features)
        for k, v in data_dict.items():
            row = [k,v['poi']]
            row = row + [v[feat] for feat in features]
            cwriter.writerow(row)

if __name__ == "__main__":
    data_dict = load_data_set()
    export_to_csv(data_dict, "csv/enron_2.csv", ana_features)