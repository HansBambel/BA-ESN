import pandas as pd
import pickle
import numpy as np

with open("squashing_scripterrors/all_squashing_errors_dataframe.p", "rb")as inputfile:
    df = pickle.load(inputfile)

print(np.min(df["averaged_error"]))
# print(df)

with open("randomprojection_scripterrors/all_randomprojection_errors_dataframe.p", "rb")as inputfile:
    df = pickle.load(inputfile)

print(np.min(df["averaged_error"]))
# print(df)

with open("squashing_scripterrors/esnError00001.p", "rb") as inputfile:
    temp = pickle.load(inputfile)
print(temp)

with open("randomprojection_scripterrors/esnError00001.p", "rb") as inputfile:
    temp = pickle.load(inputfile)
print(temp)