

import pickle

def load_data_set():
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
        return data_dict


def sort_by_feature(data_dict, feature):
    sdata_dict = sorted(data_dict.items(),key=lambda x: x[1][feature] )
    return sdata_dict

def remove_NaN(data_dict, feature):
    gdata_dict= {k:v for k, v in data_dict.items() if v[feature] !='NaN' }
    return gdata_dict

def remove_outliers(data_list, percentage):
    ldl = len(data_list)
    fpert = float(percentage)
    lbeg = int(ldl*fpert/100)
    nooutl_list = data_list[lbeg:ldl-lbeg]
    return nooutl_list

def filter_outliers(data_dict, feature, percentage):
    data_dict = remove_NaN(data_dict,feature)
    data_list = sort_by_feature(data_dict, 'salary')
    noutl_list = remove_outliers(data_list,percentage )
    noutl_dict = {k:v for (k,v) in noutl_list }
    return noutl_dict

data_dict = load_data_set()
nooutl_dict = filter_outliers(data_dict, 'salary', 10)

