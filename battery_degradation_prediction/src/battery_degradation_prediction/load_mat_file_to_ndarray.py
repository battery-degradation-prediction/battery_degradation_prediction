"""Module for loading data from .mat file and convert them into np.ndarray"""
import scipy.io
import numpy as np
import pandas as pd

def load_mat_file_to_ndarray(filepath):
    """
    """
    data_mat = scipy.io.loadmat(filepath)
    for i, (k, v) in enumerate(data_mat.items()):
        print('='*7, i, '='*7)
        print(k,v)
        #if i == 3:
        #    print(i, len(v), type(v))
        #    for id, vv in enumerate(v):
        #        for vvv in vv:
        #            print(vvv)
    #data_mat = {k:v for k, v in data_mat.items() if k[0] != '_'}
    #data = pd.DataFrame({k: pd.Series(v[0]) for k, v in data_mat.items()})
    #print(data.head())
    #data.to_csv("example.csv")
    return

def conv_matlab_to_csv(matlab_file):
    """
    This function converts a matlab file into a csv file."
    You need to pass the matlab file path as the parameter.
    """
    data = scipy.io.loadmat(matlab_file)
    cols=[col for col in data if '__' not in col]
    df=pd.DataFrame(columns=cols)
    for i in data:
        if '__' not in i :
           df[i]=(data[i]).ravel()
    #save to csv without the pandas index 
    return df.to_csv("mat3_converted_to_csv.csv", index=False)
    


#df = pd.read_csv('mat3_converted_to_csv.csv')


def main():
    filepath = "../../data/SOC_0%-60%_HalfC/PL03.mat"
    conv_matlab_to_csv(filepath)
    #load_mat_file_to_ndarray(filepath)

if __name__ == "__main__":
    main()


