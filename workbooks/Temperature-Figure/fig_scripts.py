import numpy as np
import pandas as pd
from waldo.wio.experiment import Experiment
from waldo.behavior import Behavior_Coding, Worm_Shape

def pull_blobs_from_eid(eid, path, min_time=20, min_timepoints=5000, dt=1.0):
    """
    eid - id
    pl - pathlib object to waldo directory
    min_time - (int) number of minutes track needs to be longer than in order to be considered.
    """
    #print(eid)
    #path = pl / eid / 'blob_files'
    #print(path)
    e = Experiment(fullpath=path, experiment_id=eid)
    print(e.id)
    typical_bodylength = e.typical_bodylength

    print(typical_bodylength, 'bl')
    b_list = []
    failed_dict = {}
    for i, (bid, blob) in enumerate(e.blobs()):
        blob_df = blob.df
        # print(blob_df.head())
        # print(i, len)
        # needs at least 5000 frames
        if len(blob_df) < min_timepoints:
            continue

        # needs to exist for at least min_time minutes
        t = blob_df['time']
        if t.iloc[-1] - t.iloc[0] < min_time * 60:
            continue

        # process blob
        try:
            bc = Behavior_Coding()
            bc.read_from_blob_df(blob_df)
            bc.preprocess(dt=dt)
            df = bc.df
            #del bc
            #del blob_df
            df['bl / s'] = df['speed'] / typical_bodylength
            b_list.append(df)
        except:
            failed_dict[bid] = len(blob_df)

    print(len(b_list), 'blobs match criterion')
    print(len(failed_dict), 'blobs fail criterion')
    return b_list



def aggregate_track_dfs(df_list, total_seconds):
    agg = [[] for i in range(total_seconds)]
    for df in df_list[:]:
        for i, row in df.iterrows():
            t = row['time']
            speed = row['bl / s']
            if np.isnan(t) or np.isnan(speed):
                continue
            t = int(t)
            speed = float(speed)
            agg[t].append(speed)

    times = []
    means = []
    q1, q2, q3 = [], [], []
    counts = []
    for i, points in enumerate(agg):
        count = len(points)
        if count == 0:
            continue
        times.append(i)
        means.append(np.mean(points))
        counts.append(count)
        q1.append(np.percentile(points, 25))
        q2.append(np.percentile(points, 50))
        q3.append(np.percentile(points, 75))

    #     print(len(counts))
    #     print(len(q1))
    #     print(len(q2))
    #     print(len(q3))
    #     print(len(times))
    #     print(len(means))
    #     df =  pd.DataFrame([times, means],index=['time', 'mean']).T
    #     df.loc[:, 'q1'] = q1
    #         df.loc[:, 'q1'] = q1

    df = pd.DataFrame([times, means, q1, q2, q3, counts], index=['time', 'mean', 'q1', 'q2', 'q3', 'count'])
    df = df.T
    d = pd.rolling_mean(df, window=60, center=True)
    return d