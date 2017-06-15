import pickle
import pandas as pd
import numpy as np

### Save squashing errors in array to a pickle file ###
# all = []
# for i in range(0,10000):
#     with open("squashing_scripterrors/esnError{:>05}.p".format(i), "rb") as inputfile:
#         all.append(pickle.load(inputfile))
# all = np.asarray(all)
# with open("squashing_scripterrors/all_squashing_errors.p", "wb") as outputfile:
#     pickle.dump(all, outputfile)

### Save randomprojection errors in array to a pickle file ###
# all = []
# for i in range(0,10000):
#     with open("randomprojection_scripterrors/esnError{:>05}.p".format(i), "rb") as inputfile:
#         data_dict = pickle.load(inputfile)
#         # data = [data_dict["params"]["input_leak_rate"],data_dict["params"]["input_spectral_rad"],
#         #         data_dict["params"]["output_leak_rate"],data_dict["params"]["output_spectral_rad"],
#         #         data_dict["averaged_error"],data_dict["errors"]]
#         all.append(data_dict)
# all = np.asarray(all)
# with open("randomprojection_scripterrors/all_randomprojection_errors.p", "wb") as outputfile:
#     pickle.dump(all, outputfile)


##### save in Dataframe
df = pd.DataFrame(columns=["input_leak_rate","input_spectral_rad",
                            "output_leak_rate","output_spectral_rad",
                            "errors","averaged_error"])
for i in range(0,10000):
    with open("randomprojection_100neurons_3parity_scripterrors/esnError{:>05}.p".format(i), "rb") as inputfile:
        temp = pickle.load(inputfile)
        s = pd.Series([temp["params"]["input_leak_rate"], temp["params"]["input_spectral_rad"],
                        temp["params"]["output_leak_rate"], temp["params"]["output_spectral_rad"],
                        temp["errors"], temp["averaged_error"]], index=["input_leak_rate","input_spectral_rad",
                                                                        "output_leak_rate","output_spectral_rad",
                                                                        "errors","averaged_error"])

    df = df.append(s, ignore_index=True)

df.to_pickle("randomprojection_100neurons_3parity_scripterrors/all_randomprojection_errors_dataframe.p")
