'''
    Script to identify bacteria that are far away from others. 
    These will interfere with the meshing algorithm and cause instabilities.
    Requirements:
        @ csv file with centers of bacteria as columns ['X_c', 'Y_c', 'Z_c'] in meters
'''
import numpy as np
import pandas as pd
import sklearn.cluster as scl

# ===
# VARIABLES
# ===
fname = 'capsules.csv'
outf = 'capsules'
write = True
eps = 4e-6  # Variable for clustering, ~ expected distance between bacs

# ===
# READ DATAFRAME
# ===
df = pd.read_csv(fname)
# df = df[df.time == np.max(df.time)]  # Uncomment when there are multiple time steps

# ===
# CLUSTER TO IDENTIFY OUTLIERS WITH DBSCAN
# try with min_samples = 5 if outliers are clustered together instead of at -1
# ===
cl = scl.DBSCAN(eps=).fit(df[['X_c', 'Y_c',
                                  'Z_c']].to_numpy())
df['clust_id'] = cl.labels_
dfo = df[df['clust_id'] == -1]
dfno = df[df['clust_id'] != -1]

# ===
# WRITE RESULTS
# ===
if write:
    df.to_csv(f'{outf}_w_outliers.csv', index=False)
    dfo.to_csv(f'{outf}_outliers.csv', index=False)
    dfno.to_csv(f'{outf}_wo_outliers.csv', index=False)