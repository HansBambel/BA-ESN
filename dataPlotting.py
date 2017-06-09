import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("squashing_scripterrors/all_squashing_errors_dataframe.p", "rb")as inputfile:
    df = pd.DataFrame(pickle.load(inputfile))

# # print(np.min(df["averaged_error"]))
# min_row = df.loc[df["averaged_error"]==np.min(df["averaged_error"])]
# print(min_row[["input_leak_rate", "input_spectral_rad", "output_leak_rate", "output_spectral_rad"]])
#
#
# # get inputESN parameter for lowest error
# grouped = df.loc[(df["input_leak_rate"]==min_row.iloc[0]["input_leak_rate"]) &
#                  (df["input_spectral_rad"]==min_row.iloc[0]["input_spectral_rad"])]
#
# ### TODO get rid of outlier or use median
# print("Errors of max: ",grouped.loc[grouped["averaged_error"]==np.max(grouped["averaged_error"])].iloc[0]["errors"])
# print("Errors of min",grouped.loc[grouped["averaged_error"]==np.min(grouped["averaged_error"])].iloc[0]["errors"])
#
# grouped = grouped.pivot(index="output_leak_rate", columns="output_spectral_rad",values="averaged_error")
# X=grouped.index.values
# Y=grouped.columns.values
# Z=grouped.values
# xx,yy = np.meshgrid(X,Y)
# plt.contourf(xx,yy,Z.T,30)
# plt.colorbar()
# plt.show()

## TODO expected error given E[error|l1,R1] und E[error|L2,R2] (marginalize)
expected_error = df[["input_leak_rate","input_spectral_rad","averaged_error"]].groupby(["input_leak_rate","input_spectral_rad"], as_index=False)
# print(expected_error.mean())
mean_e_e=expected_error.mean()
temp=mean_e_e["averaged_error"].to_dense().values.reshape(10,10)
xx = mean_e_e["input_leak_rate"].reshape(10,10)
yy = mean_e_e["input_spectral_rad"].reshape(10,10)
plt.contourf(xx,yy,temp,30)
plt.colorbar()
plt.show()

expected_error = df[["input_leak_rate","input_spectral_rad","averaged_error"]].groupby(["input_leak_rate","input_spectral_rad"], as_index=False)
# print(expected_error.mean())
mean_e_e=expected_error.mean()
temp=mean_e_e["averaged_error"].to_dense().values.reshape(10,10)
xx = mean_e_e["input_leak_rate"].reshape(10,10)
yy = mean_e_e["input_spectral_rad"].reshape(10,10)
plt.contourf(xx,yy,temp,30)
plt.colorbar()
plt.show()
# ############# Randomprojection
# with open("randomprojection_scripterrors/all_randomprojection_errors_dataframe.p", "rb")as inputfile:
#     df = pd.DataFrame(pickle.load(inputfile))
#
# # print(np.min(df["averaged_error"]))
# min_row = df.loc[df["averaged_error"]==np.min(df["averaged_error"])]
# print(min_row[["input_leak_rate", "input_spectral_rad", "output_leak_rate", "output_spectral_rad"]])
#
#
# # get inputESN parameter for lowest error
# grouped = df.loc[(df["input_leak_rate"]==min_row.iloc[0]["input_leak_rate"]) &
#                  (df["input_spectral_rad"]==min_row.iloc[0]["input_spectral_rad"])]
#
# ### TODO get rid of outlier
# print("Errors of max: ",grouped.loc[grouped["averaged_error"]==np.max(grouped["averaged_error"])].iloc[0]["errors"])
# print("Errors of min",grouped.loc[grouped["averaged_error"]==np.min(grouped["averaged_error"])].iloc[0]["errors"])
#
# grouped = grouped.pivot(index="output_leak_rate", columns="output_spectral_rad",values="averaged_error")
# X=grouped.index.values
# Y=grouped.columns.values
# Z=grouped.values
# xx,yy = np.meshgrid(X,Y)
# plt.contourf(xx,yy,Z.T,30)
# plt.colorbar()
# plt.show()

