import pickle
import pandas as pd

# all = []
# for i in range(0,10000):
#     with open("squashing_scripterrors/esnError{:>05}.p".format(i), "rb") as inputfile:
#         all.append(pickle.load(inputfile))
#
# print(len(all))
# # print(range(0,10000))
# with open("squashing_scripterrors/all_squashing_errors.p", "wb") as outputfile:
#     pickle.dump(all, outputfile)

# with open("randomprojection_scripterrors/all_randomprojection_errors.p", "rb") as file:
#     all_errors = pickle.load(file)

# with open("squashing_scripterrors/all_squashing_errors.p", "rb") as file:
#     all_errors = pickle.load(file)

# with open("squashing_scripterrors/esnError00001.p", "rb") as inputfile:
#     temp = pickle.load(inputfile)
# print(temp)
# print(temp["params"]["input_leak_rate"])

# df = pd.DataFrame([temp["params"]["input_leak_rate"], temp["params"]["input_spectral_rad"],
#                    temp["params"]["output_leak_rate"], temp["params"]["output_spectral_rad"],
#                    temp["errors"], temp["averaged_error"]])

# df = pd.DataFrame({"input_leak_rate":temp["params"]["input_leak_rate"], "input_spectral_rad":temp["params"]["input_spectral_rad"],
#                    "output_leak_rate":temp["params"]["output_leak_rate"], "output_spectral_rad":temp["params"]["output_spectral_rad"],
#                    "averaged_error":temp["averaged_error"]})

df = pd.DataFrame(columns=["input_leak_rate","input_spectral_rad",
                            "output_leak_rate","output_spectral_rad",
                            "errors","averaged_error"])
for i in range(0,10000):
    with open("squashing_scripterrors/esnError{:>05}.p".format(i), "rb") as inputfile:
        temp = pickle.load(inputfile)
        s = pd.Series([temp["params"]["input_leak_rate"], temp["params"]["input_spectral_rad"],
                        temp["params"]["output_leak_rate"], temp["params"]["output_spectral_rad"],
                        temp["errors"], temp["averaged_error"]], index=["input_leak_rate","input_spectral_rad",
                                                                        "output_leak_rate","output_spectral_rad",
                                                                        "errors","averaged_error"])

    df = df.append(s, ignore_index=True)

df.to_pickle("squashing_scripterrors/all_squashing_errors_dataframe.p")
