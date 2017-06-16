import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("squashing_scripterrors/all_squashing_errors_dataframe.p", "rb")as inputfile:
    df = pd.DataFrame(pickle.load(inputfile))

# print(np.min(df["averaged_error"]))
# min_row = df.loc[df["averaged_error"]==np.min(df["averaged_error"])]
# print(min_row[["input_leak_rate", "input_spectral_rad", "output_leak_rate", "output_spectral_rad"]])
#
#
# # get inputESN parameter for lowest error
# grouped = df.loc[(df["input_leak_rate"]==min_row.iloc[0]["input_leak_rate"]) &
#                  (df["input_spectral_rad"]==min_row.iloc[0]["input_spectral_rad"])]
#
# ### get rid of outlier or use median
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
'''
## expected error given E[error|l1,R1] (marginalize)
expected_error_given_input = df[["input_leak_rate","input_spectral_rad","averaged_error"]].groupby(["input_leak_rate","input_spectral_rad"], as_index=False)
# print(expected_error.mean())
plt.figure(figsize=(14,5))
plt.subplot(121)
mean_e_e=expected_error_given_input.mean()
temp=mean_e_e["averaged_error"].to_dense().values.reshape(10,10)
xx = mean_e_e["input_leak_rate"].reshape(10,10)
yy = mean_e_e["input_spectral_rad"].reshape(10,10)
plt.contourf(xx,yy,temp,30)
plt.colorbar()
plt.title("Expected error given input parameters")

## expected error given E[error|l1,R1] (marginalize)
expected_error_given_output = df[["output_leak_rate","output_spectral_rad","averaged_error"]].groupby(["output_leak_rate","output_spectral_rad"], as_index=False)
# print(expected_error.mean())
plt.subplot(122)
mean_e_e=expected_error_given_output.mean()
temp=mean_e_e["averaged_error"].to_dense().values.reshape(10,10)
xx = mean_e_e["output_leak_rate"].reshape(10,10)
yy = mean_e_e["output_spectral_rad"].reshape(10,10)
plt.contourf(xx,yy,temp,30)
plt.colorbar()
plt.title("Expected error given output parameters")
plt.show()
'''

# ############# Randomprojection

# with open("randomprojection_scripterrors_30/all_50_randomprojection_errors_dataframe.p", "rb")as inputfile:
with open("randomprojection_100neurons_3parity_scripterrors/all_randomprojection_errors_dataframe.p", "rb")as inputfile:
    df = pd.DataFrame(pickle.load(inputfile))
#
# # print(np.min(df["averaged_error"]))
min_row = df.loc[df["averaged_error"]==np.min(df["averaged_error"])]
# print(min_row[["input_leak_rate", "input_spectral_rad", "output_leak_rate", "output_spectral_rad"]])

## expected error given E[error|l1,R1] (marginalize)
expected_error_given_input = df[["input_leak_rate","input_spectral_rad","averaged_error"]].groupby(["input_leak_rate","input_spectral_rad"], as_index=False)
# print(expected_error.mean())
plt.figure(figsize=(14,5))
plt.subplot(121)
mean_e_e=expected_error_given_input.mean()
temp=mean_e_e["averaged_error"].to_dense().values.reshape(10,10)
xx = mean_e_e["input_leak_rate"].reshape(10,10)
yy = mean_e_e["input_spectral_rad"].reshape(10,10)
plt.contourf(xx,yy,temp,30)
plt.plot(min_row["input_leak_rate"],min_row["input_spectral_rad"],"wx")
plt.colorbar()
plt.title("Expected error given input parameters")

## expected error given E[error|l1,R1] (marginalize)
expected_error_given_output = df[["output_leak_rate","output_spectral_rad","averaged_error"]].groupby(["output_leak_rate","output_spectral_rad"], as_index=False)
# print(expected_error.mean())
plt.subplot(122)
mean_e_e=expected_error_given_output.mean()
temp=mean_e_e["averaged_error"].to_dense().values.reshape(10,10)
xx = mean_e_e["output_leak_rate"].reshape(10,10)
yy = mean_e_e["output_spectral_rad"].reshape(10,10)
plt.contourf(xx,yy,temp,30)
plt.plot(min_row["output_leak_rate"],min_row["output_spectral_rad"],"wx")
plt.colorbar()
plt.title("Expected error given output parameters")
plt.show()

##### given best input params plot errors of second esn
plt.figure(figsize=(14,5))
plt.subplot(121)
# # best_first_params = df[["input_leak_rate"==min_row["input_leak_rate"], "input_spectral_rad"==min_row["input_spectral_rad"],"averaged_error"]].groupby(["output_leak_rate","output_spectral_rad"], as_index=False)
best_first_params = df.loc[(df["input_leak_rate"]==min_row.iloc[0]["input_leak_rate"]) &
                           (df["input_spectral_rad"]==min_row.iloc[0]["input_spectral_rad"])]
# mean_best_first= best_first_params.mean()
# print(best_first_params)
zz = best_first_params["averaged_error"].to_dense().values.reshape(10,10)
xx = best_first_params["output_leak_rate"].reshape(10,10)
yy = best_first_params["output_spectral_rad"].reshape(10,10)
plt.contourf(xx,yy,zz,30)
plt.plot(min_row["output_leak_rate"],min_row["output_spectral_rad"],"wx")
plt.colorbar()
plt.title("Expected error given optimal first params")

######### given best output params plot errors of first esn
plt.subplot(122)
# # best_first_params = df[["input_leak_rate"==min_row["input_leak_rate"], "input_spectral_rad"==min_row["input_spectral_rad"],"averaged_error"]].groupby(["output_leak_rate","output_spectral_rad"], as_index=False)
best_second_params = df.loc[(df["output_leak_rate"]==min_row.iloc[0]["output_leak_rate"]) &
                           (df["output_spectral_rad"]==min_row.iloc[0]["output_spectral_rad"])]
# mean_best_first= best_first_params.mean()
# print(best_second_params)
zz = best_second_params["averaged_error"].to_dense().values.reshape(10,10)
xx = best_second_params["input_leak_rate"].reshape(10,10)
yy = best_second_params["input_spectral_rad"].reshape(10,10)
plt.contourf(xx,yy,zz,30)
plt.plot(min_row["input_leak_rate"],min_row["input_spectral_rad"],"wx")
plt.colorbar()
plt.title("Expected error given optimal second params")
plt.show()