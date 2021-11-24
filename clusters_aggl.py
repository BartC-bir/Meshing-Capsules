'''
    Script to identify bacteria that are far away from others. 
    These will interfere with the meshing algorithm and cause instabilities.
    Requirements:
        @ csv file with centers of bacteria, without outliers (after identify_outliers script)
    Carefull! This might write out a lot of files 
'''
import numpy as np
import pandas as pd
import sklearn.cluster as scl
from tqdm import tqdm

# ===
# VARIABLES
# ===
fname = 'capsules_wo_outliers.csv'
thr = 5e-4  # Threshold for meshing
write = True
outdir = '.'  # Directory to write xyz files to

# ===
# READ DATA
# ===
df = pd.read_csv(fname)
# df = df[df.time == np.max(df.time)]  # Uncomment when there are multiple time steps


# ===
# GENERAL CLUSTERING
# ===
cl = scl.AgglomerativeClustering(n_clusters=None,
                                 distance_threshold=thr,
                                 linkage='ward').fit(df[['X_c', 'Y_c',
                                                         'Z_c']].to_numpy())

# === 
# WRITE NEW LABELS
# ===
if write:
    df['clust_id'] = cl.labels_
    df.to_csv('capsules_wo_outliers.csv', index=False)

# ===
# WRITE THE DATA AS XYZ FILES, CAN BE USED WITH OHTER MESHING ALGO'S AS WELL
# ===
for li in tqdm(np.unique(np.array(df['clust_id'])), desc='Writing clusters'):
    xyz = np.vstack((df[['X_0', 'Y_0', 'Z_0']][np.array(df['clust_id']) == li],
                     df[['X_1', 'Y_1',
                         'Z_1']][np.array(df['clust_id']) == li]))
    np.savetxt(f'{outdir}/capsules_{li:03g}.xyz', xyz)
