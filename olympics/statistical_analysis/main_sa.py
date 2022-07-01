import os
from modules import pose_estimation_openpose
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

col_names = list((pose_estimation.keypointsMapping[pose_estimation.POSE_PAIRS[t][0]] + " to " +
                  pose_estimation.keypointsMapping[pose_estimation.POSE_PAIRS[t][1]]) for t in
                 range(len(pose_estimation.POSE_PAIRS)))
col_names = col_names[:-2]
# restore pandas dataframe that was squished into a csv cell as string
def restore_df(str_in):
    str_in = str_in[1:-1]
    str_in = str_in.replace("\n", "")\
        .replace("[[", "")\
        .replace("]]", "")\
        .replace("] [ ", "__")\
        .replace("]","")\
        .replace("[", "")\
        .split("__")
    out = []
    for i in range(len(str_in)):
        row = str_in[i].split(",")
        row = row[0].split(" ")
        for elem in row:
            if elem in ['"', '', ' ', "''", "''",""]:
                row.remove(elem)
        out.append(row)
    for i in range(len(out)):
        out[i] = [x for x in out[i] if x]

    for x in out:
        del x[0]

    return pd.DataFrame(out)


# input: scenes_annotated_01.csv from detector/main_va.py
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))
scenes_annotated_path = os.path.join(ROOT_DIR, '0_resources', 'scenes_annotated_02.csv')
# avg distance and avg synchronization per performance

scenes_annotated = pd.read_csv(scenes_annotated_path)

# restore dataframes
# outputs dataframe where first element of each row is the frame, followed by the data points of the timeseries
# get avg distance and avg synchronization per performance
scenes_annotated["synchrony_avg"] = ""
#for idx, row in scenes_annotated.iterrows():

synchrony_avg = []
# creates avg synchrony value over all frames of a video snippet, but for the different body parts individually
for idx, row in scenes_annotated.iterrows():
    temp = restore_df(row.synchrony_timeseries)
    temp = temp.apply(pd.to_numeric, errors='coerce')
    temp = temp.replace(-1, np.NaN)
    temp = temp.mean(axis=0)
    synchrony_avg.append(temp)

df = pd.DataFrame(synchrony_avg)
df["SPTSS"] = scenes_annotated.SPTSS
df["Team"] = scenes_annotated.Team
df2 = df.groupby(["Team"]).mean()

for i in range(17):
    df.columns.values[i] = col_names[i]


rho = df2.corr()
pval = df2.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t]))
rho.round(2).astype(str) + p

print("hi")
#pyplot.scatter(data_grouped.SPTSS, data_grouped)
#pyplot.show()
    #23.974
# correlation
