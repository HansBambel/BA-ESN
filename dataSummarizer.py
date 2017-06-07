import pickle
import pandas as pd

### Save squashing errors in array to a pickle file ###
# all = []
# for i in range(0,10000):
#     with open("squashing_scripterrors/esnError{:>05}.p".format(i), "rb") as inputfile:
#         all.append(pickle.load(inputfile))
# with open("squashing_scripterrors/all_squashing_errors.p", "wb") as outputfile:
#     pickle.dump(all, outputfile)

### Save randomprojection errors in array to a pickle file ###
# all = []
# for i in range(0,10000):
#     with open("randomprojection_scripterrors/esnError{:>05}.p".format(i), "rb") as inputfile:
#         all.append(pickle.load(inputfile))
# with open("randomprojection_scripterrors/all_randomprojection_errors.p", "wb") as outputfile:
#     pickle.dump(all, outputfile)

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
