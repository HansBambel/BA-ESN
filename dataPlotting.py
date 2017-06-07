import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("squashing_scripterrors/all_squashing_errors_dataframe.p", "rb")as inputfile:
    df = pd.DataFrame(pickle.load(inputfile))

# print(np.min(df["averaged_error"]))
min_row = df.loc[df["averaged_error"]==np.min(df["averaged_error"])]
print(min_row.iloc[0]["input_leak_rate"])
print(min_row.iloc[0]["input_spectral_rad"])


# get inputESN parameter for lowest error
grouped = df.loc[(df["input_leak_rate"]==min_row.iloc[0]["input_leak_rate"]) &
                 (df["input_spectral_rad"]==min_row.iloc[0]["input_spectral_rad"])]

### TODO get rid of outlier
print(grouped.loc[grouped["averaged_error"]==np.max(grouped["averaged_error"])].iloc[0]["errors"])

grouped = grouped.pivot(index="output_leak_rate", columns="output_spectral_rad",values="averaged_error")
X=grouped.index.values
Y=grouped.columns.values
Z=grouped.values
xx,yy = np.meshgrid(X,Y)
plt.contourf(xx,yy,Z.T,30)
plt.colorbar()
plt.show()

############# Randomprojection
# with open("randomprojection_scripterrors/all_randomprojection_errors_dataframe.p", "rb")as inputfile:
#     df = pd.DataFrame(pickle.load(inputfile))
#
# # print(np.min(df["averaged_error"]))
# min_row = df.loc[df["averaged_error"]==np.min(df["averaged_error"])]
# print(min_row.iloc[0]["input_leak_rate"])
# print(min_row.iloc[0]["input_spectral_rad"])
#
#
# # get inputESN parameter for lowest error
# grouped = df.loc[(df["input_leak_rate"]==min_row.iloc[0]["input_leak_rate"]) &
#                  (df["input_spectral_rad"]==min_row.iloc[0]["input_spectral_rad"])]
#
# ### TODO get rid of outlier
# print(grouped.loc[grouped["averaged_error"]==np.max(grouped["averaged_error"])].iloc[0]["errors"])
#
# grouped = grouped.pivot(index="output_leak_rate", columns="output_spectral_rad",values="averaged_error")
# X=grouped.index.values
# Y=grouped.columns.values
# Z=grouped.values
# xx,yy = np.meshgrid(X,Y)
# plt.contourf(xx,yy,Z.T,30)
# plt.colorbar()
# plt.show()

