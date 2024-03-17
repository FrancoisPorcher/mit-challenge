import pandas as pd
from fastcore.basics import Path
import numpy as np
import os
from node import Node
from datetime import datetime, timedelta
from pathlib import Path
# from statsmodels.tsa.seasonal import seasonal_decompose

# Function to prepare the data in a tabular format


# def decompose(series, period):
#     decomposition = seasonal_decompose(
#         series, model='additive', period=period)
#     return decomposition.resid.ffill().bfill()


def tabularize_data(data_dir, feature_cols, ground_truth=None, lag_steps=1, add_heurestic=False, nb_of_ex=1000000):
    merged_data = pd.DataFrame()
    test_data = Path(data_dir).glob('*.csv')

    # Check if test_data is empty
    if not test_data:
        raise ValueError(f'No csv files found in {data_dir}')
    starttime = datetime.fromisoformat("2023-01-01 00:00:00+00:00")
    endtime = datetime.fromisoformat("2023-07-01 00:00:00+00:00")
    for data_file in list(test_data)[:nb_of_ex]:
        data_df = pd.read_csv(data_file)
        data_df['ObjectID'] = int(data_file.stem)
        data_df['TimeIndex'] = range(len(data_df))
        data_df['Time36'] = pd.Series(
            np.tile(np.arange(1, 36+1), len(data_df))[:len(data_df)])

        new_feature_cols = list(feature_cols)  # Create a copy of feature_cols

        for time_windows in [6, 12, 12*3, 12*7, 12*14]:
            time_name = f'time_{time_windows}'
            data_df[time_name] = pd.Series(
                np.tile(np.arange(1, time_windows+1), len(data_df))[:len(data_df)])
            new_feature_cols.append(time_name)

        if add_heurestic:
            data_df = add_baseline_heuristic(data_df, int(
                data_file.stem), starttime, endtime, feature_cols)

        if ground_truth is None:
            merged_df = data_df
        else:
            ground_truth_object = ground_truth[ground_truth['ObjectID']
                                               == data_df['ObjectID'][0]].copy()
            # Separate the 'EW' and 'NS' types in the ground truth
            ground_truth_EW = ground_truth_object[ground_truth_object['Direction'] == 'EW'].copy(
            )
            ground_truth_NS = ground_truth_object[ground_truth_object['Direction'] == 'NS'].copy(
            )

            # Create 'EW' and 'NS' labels and fill 'unknown' values
            ground_truth_EW['EW'] = ground_truth_EW['Node'] + \
                '-' + ground_truth_EW['Type']
            ground_truth_NS['NS'] = ground_truth_NS['Node'] + \
                '-' + ground_truth_NS['Type']
            ground_truth_EW.drop(
                ['Node', 'Type', 'Direction'], axis=1, inplace=True)
            ground_truth_NS.drop(
                ['Node', 'Type', 'Direction'], axis=1, inplace=True)

            # Merge the input data with the ground truth
            merged_df = pd.merge(data_df,
                                 ground_truth_EW.sort_values('TimeIndex'),
                                 on=['TimeIndex', 'ObjectID'],
                                 how='left')
            merged_df = pd.merge_ordered(merged_df,
                                         ground_truth_NS.sort_values(
                                             'TimeIndex'),
                                         on=['TimeIndex', 'ObjectID'],
                                         how='left')

            # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
            # merged_df['EW'].ffill(inplace=True)
            # merged_df['NS'].ffill(inplace=True)

        merged_data = pd.concat([merged_data, merged_df])

    # envelope feature
    enveloppe_feature = []
    for factor in ["Latitude (deg)", "Longitude (deg)",  "Altitude (m)"]:
        enveloppe_name = f'{factor}_enveloppe'
        added_data = merged_data.groupby('ObjectID')[factor].apply(lambda win: win.rolling(3*12, center=True, min_periods=0).max()) - \
            merged_data.groupby('ObjectID')[factor].apply(
                lambda win: win.rolling(3*12, center=True, min_periods=0).min())
        added_data = added_data.rename(enveloppe_name)
        added_data.index = merged_data.index
        new_feature_cols.append(enveloppe_name)
        enveloppe_feature.append(added_data)

    # do it now to compute diff on it
    merged_data = pd.concat([merged_data] + enveloppe_feature, axis=1)

    # Create lagged features for each column in feature_cols
    lagged_features = []
    list_lag = list(range(-lag_steps, lag_steps+1))
    list_lag.remove(0)
    for col in feature_cols:
        for i in list_lag:
            lag_col_name = f'{col}_lag_{i}'
            lagged_features.append(merged_data.groupby('ObjectID')[
                col].shift(i).ffill().bfill().rename(lag_col_name))
            # Add the lagged feature to new_feature_cols
            new_feature_cols.append(lag_col_name)

    diff_features = []
    for col in ["Eccentricity",  "Semimajor Axis (m)",  "Inclination (deg)",  "RAAN (deg)", "Argument of Periapsis (deg)", "True Anomaly (deg)",  "Latitude (deg)",
                "Longitude (deg)",   "Altitude (m)", "Latitude (deg)_enveloppe", "Longitude (deg)_enveloppe",  "Altitude (m)_enveloppe"]:
        for i in [-2, -1, 1, 2]:
            diff_col_name = f'{col}_diff_{i}'
            diff_features.append(merged_data.groupby('ObjectID')[col].diff(
                i).ffill().bfill().rename(diff_col_name))
            # Add the new feature to new_feature_cols
            new_feature_cols.append(diff_col_name)

    pct_features = []
    for col in ['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)',]:
        for i in [-1, 1]:
            pct_col_name = f'{col}_pct_change_{i}'
            pct_features.append(merged_data.groupby('ObjectID')[col].pct_change(
                i).ffill().bfill().rename(pct_col_name))
            # Add the new feature to new_feature_cols
            new_feature_cols.append(pct_col_name)

    rolling_features = []
    for variable in ["Eccentricity",  "Semimajor Axis (m)", "Inclination (deg)",  "RAAN (deg)", "Argument of Periapsis (deg)", "True Anomaly (deg)", "Altitude (m)",
                     "Vx (m/s)",  "Vy (m/s)", "Vz (m/s)"]:
        rolling_std_name = f'{variable}_rolling_std_12'
        added_data = merged_data.groupby('ObjectID')[variable].apply(
            lambda win: win.rolling(12, center=True).std().ffill().bfill()).rename(rolling_std_name)
        added_data.index = merged_data.index
        rolling_features.append(added_data)
        new_feature_cols.append(rolling_std_name)
        rolling_mean_name = f'{variable}_rolling_mean_12'
        added_data = merged_data.groupby('ObjectID')[variable].apply(
            lambda win: win.rolling(12, center=True).mean().ffill().bfill()).rename(rolling_mean_name)
        added_data.index = merged_data.index
        rolling_features.append(added_data)
        new_feature_cols.append(rolling_mean_name)

    charac_features = []
    for factor in ["Eccentricity", "Semimajor Axis (m)", "Inclination (deg)", "RAAN (deg)", "Argument of Periapsis (deg)", "True Anomaly (deg)", "Altitude (m)"]:
        charach_name = f'{factor}_mean_charac'
        added_data = pd.merge(merged_data[['ObjectID', 'Timestamp']],
                              merged_data.groupby('ObjectID')[factor].mean().rename(charach_name), on='ObjectID', how='left')[charach_name]
        added_data.index = merged_data.index
        charac_features.append(added_data)
        new_feature_cols.append(charach_name)

    for factor in ["X (m)", "Y (m)", "Z (m)", "Vx (m/s)",  "Vy (m/s)", "Vz (m/s)", "Eccentricity", "Altitude (m)", "Inclination (deg)"]:
        charach_name = f'{factor}_std_charac'
        added_data = pd.merge(merged_data[['ObjectID', 'Timestamp']],
                              merged_data.groupby('ObjectID')[factor].std().rename(charach_name), on='ObjectID', how='left')[charach_name]
        added_data.index = merged_data.index
        charac_features.append(added_data)
        new_feature_cols.append(charach_name)

    # seasonal_features = []
    # for factor in ["Eccentricity", "Altitude (m)", "Inclination (deg)"]:
    #     seasonal_name = f'{factor}_12_resid'
    #     added_data = merged_data.groupby('ObjectID')[
    #                              factor].apply(decompose, period=12).rename(seasonal_name)
    #     added_data.index = merged_data.index
    #     seasonal_features.append(added_data)
    #     new_feature_cols.append(seasonal_name)

    # Add the lagged features to the DataFrame all at once
    merged_data = pd.concat(
        [merged_data] + lagged_features + diff_features + pct_features + rolling_features + charac_features, axis=1)  # + seasonal_features

    if add_heurestic:
        dummies_ew = pd.get_dummies(merged_data[['EW_baseline_heuristic']])
        dummies_ns = pd.get_dummies(merged_data[['NS_baseline_heuristic']])
        dummies_ew_ffill = pd.get_dummies(
            merged_data[['EW_baseline_heuristic_ffill']])
        dummies_ns_ffill = pd.get_dummies(
            merged_data[['NS_baseline_heuristic_ffill']])
        merged_data = pd.concat(
            [merged_data, dummies_ew, dummies_ns, dummies_ew_ffill, dummies_ns_ffill], axis=1)
        new_feature_cols = new_feature_cols + \
            list(dummies_ew.columns) + list(dummies_ns.columns) + \
            list(dummies_ew_ffill.columns) + list(dummies_ns_ffill.columns)
        for i in ['EW_baseline_heuristic_AD-NK', 'EW_baseline_heuristic_ID-NK', 'EW_baseline_heuristic_IK-CK', 'EW_baseline_heuristic_IK-EK',
                  'EW_baseline_heuristic_SS-CK', 'EW_baseline_heuristic_SS-EK', 'EW_baseline_heuristic_SS-NK', 'NS_baseline_heuristic_ID-NK',
                  'NS_baseline_heuristic_IK-CK', 'NS_baseline_heuristic_IK-EK', 'NS_baseline_heuristic_SS-CK', 'NS_baseline_heuristic_SS-EK',
                  'NS_baseline_heuristic_SS-NK']:
            if (i in new_feature_cols) == False:
                merged_data[i] = 0
                new_feature_cols.append(i)
        for i in ['EW_baseline_heuristic_ffill_AD-NK', 'EW_baseline_heuristic_ffill_ID-NK', 'EW_baseline_heuristic_ffill_IK-CK',
                  'EW_baseline_heuristic_ffill_IK-EK',
                  'EW_baseline_heuristic_ffill_SS-CK', 'EW_baseline_heuristic_ffill_SS-EK', 'EW_baseline_heuristic_ffill_SS-NK',
                  'NS_baseline_heuristic_ffill_ID-NK',
                  'NS_baseline_heuristic_ffill_IK-CK', 'NS_baseline_heuristic_ffill_IK-EK', 'NS_baseline_heuristic_ffill_SS-CK',
                  'NS_baseline_heuristic_ffill_SS-EK',
                  'NS_baseline_heuristic_ffill_SS-NK']:
            if (i in new_feature_cols) == False:
                merged_data[i] = 0
                new_feature_cols.append(i)

    return merged_data, new_feature_cols


def convert_classifier_output(classifier_output):
    # Split the 'Predicted_EW' and 'Predicted_NS' columns into
    # 'Node' and 'Type' columns
    ew_df = classifier_output[['TimeIndex', 'ObjectID', 'Predicted_EW']].copy()
    ew_df[['Node', 'Type']] = ew_df['Predicted_EW'].str.split('-', expand=True)
    ew_df['Direction'] = 'EW'
    ew_df.drop(columns=['Predicted_EW'], inplace=True)

    ns_df = classifier_output[['TimeIndex', 'ObjectID', 'Predicted_NS']].copy()
    ns_df[['Node', 'Type']] = ns_df['Predicted_NS'].str.split('-', expand=True)
    ns_df['Direction'] = 'NS'
    ns_df.drop(columns=['Predicted_NS'], inplace=True)

    # Concatenate the processed EW and NS dataframes
    final_df = pd.concat([ew_df, ns_df], ignore_index=True)

    # Sort dataframe based on 'ObjectID', 'Direction' and 'TimeIndex'
    final_df.sort_values(['ObjectID', 'Direction', 'TimeIndex'], inplace=True)

    # Apply the function to each group of rows with the same 'ObjectID' and 'Direction'
    groups = final_df.groupby(['ObjectID', 'Direction'])
    keep = groups[['Node', 'Type']].apply(
        lambda group: group.shift() != group).any(axis=1)

    # Filter the DataFrame to keep only the rows we're interested in
    keep.index = final_df.index
    final_df = final_df[keep]

    # Reset the index and reorder the columns
    final_df = final_df.reset_index(drop=True)
    final_df = final_df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']]
    final_df = final_df.sort_values(['ObjectID', 'TimeIndex', 'Direction'])

    return final_df


class index_dict:
    def __init__(self):
        self.times = self.IDADIK()
        self.indices = []
        self.AD_dex = []
        self.modes = self.mode()

    class IDADIK:
        def __init__(self):
            self.ID = []
            self.AD = []
            self.IK = []

    class mode:
        def __init__(self):
            self.SK = []
            self.end = []


def detect_ew_pol_nodes(data, satcat, starttime, endtime):

    # Get std for longitude over a 24 hours window
    lon_std = []
    steps_per_day = 12
    lon_was_baseline = True
    lon_baseline = 0.03
    detected = index_dict()
    for i in range(len(data["Longitude (deg)"])):
        if i <= steps_per_day:
            lon_std.append(np.std(data["Longitude (deg)"][0:steps_per_day]))
        else:
            lon_std.append(np.std(data["Longitude (deg)"][i-steps_per_day:i]))

    ssEW = Node(satcat=satcat,
                t=starttime,
                index=0,
                ntype="SS",
                signal="EW")

    # Run LS detection
    # if at least 1 day has elapsed since t0
    for i in range(steps_per_day+1, len(lon_std)-steps_per_day):
        max_lon_std_24h = np.max(lon_std[i-steps_per_day:i])
        min_lon_std_24h = np.min(lon_std[i-steps_per_day:i])
        A = np.abs(max_lon_std_24h-min_lon_std_24h)/2
        th_ = 0.95*A

        # ID detection
        # if sd is elevated & last sd was at baseline
        if (lon_std[i] > lon_baseline) & lon_was_baseline:
            # mean of previous day's longitudes
            before = np.mean(data["Longitude (deg)"][i-steps_per_day:i])
            # mean of next day's longitudes
            after = np.mean(data["Longitude (deg)"][i:i+steps_per_day])
            # if not temporary noise, then real ID
            # if means are different
            if np.abs(before-after) > 0.3:
                # the sd is not yet back at baseline
                lon_was_baseline = False
                index = i
                if i < steps_per_day+2:
                    ssEW.mtype = "NK"
                else:
                    detected.times.ID.append(starttime+timedelta(hours=i*2))
        # IK detection
        # elif sd is not elevated and drift has already been initialized
        elif (lon_std[i] <= lon_baseline) & (not lon_was_baseline):
            # toggle end-of-drift boolean
            drift_ended = True
            # for the next day, check...
            for j in range(steps_per_day):
                # if the longitude changes from the current value
                if np.abs(data["Longitude (deg)"][i]-data["Longitude (deg)"][i+j]) > 0.3:
                    # the drift has not ended
                    drift_ended = False
            if drift_ended:                                                 # if the drift has ended
                # the sd is back to baseline
                lon_was_baseline = True
                # save tnow as end-of-drift
                detected.times.IK.append(starttime+timedelta(hours=i*2))
                # save indices for t-start & t-end
                detected.indices.append([index, i])

        # Last step
        elif (i == (len(lon_std)-steps_per_day-1))\
                & (not lon_was_baseline):
            detected.times.IK.append(starttime+timedelta(hours=i*2))
            detected.indices.append([index, i])

        # AD detection
        # elif sd is elevated and drift has already been initialized
        elif ((lon_std[i]-max_lon_std_24h > th_) or (min_lon_std_24h-lon_std[i] > th_)) & (not lon_was_baseline):
            if i >= steps_per_day+3:
                detected.times.AD.append(starttime+timedelta(hours=i*2))
                detected.AD_dex.append(i)
    return detected, ssEW, lon_std


def filter_and_merge_nodes(data, detected, ssEW, lon_std, satcat, starttime, endtime, steps_per_day):
    nodes = []
    filtered = index_dict()
    longitudes = data["Longitude (deg)"]
    inclinations = data["Inclination (deg)"]

    def add_node(n):
        nodes[len(nodes)-1].char_mode(
            next_index=n.index,
            lons=longitudes,
            incs=inclinations
        )
        if n.type == "AD":
            nodes[len(nodes)-1].mtype = "NK"

        if (nodes[len(nodes)-1].mtype != "NK"):
            filtered.indices.append([nodes[len(nodes)-1].index, n.index])
            filtered.modes.SK.append(nodes[len(nodes)-1].mtype)
            stop_NS = True if n.type == "ID" else False
            filtered.modes.end.append(stop_NS)
        nodes.append(n)

    es = Node(satcat=satcat,
              t=endtime,
              index=len(data["Longitude (deg)"])-1,
              ntype="ES",
              signal="ES",
              mtype="ES")

    toggle = True
    nodes.append(ssEW)
    if len(detected.times.IK) == 1:
        if len(detected.times.ID) == 1:
            # keep the current ID
            filtered.times.ID.append(detected.times.ID[0])
            ID = Node(satcat,
                      detected.times.ID[0],
                      index=detected.indices[0][0],
                      ntype='ID',
                      lon=longitudes[detected.indices[0][0]],
                      signal="EW")
            add_node(ID)
        filtered.times.IK.append(detected.times.IK[0])
        IK = Node(satcat,
                  detected.times.IK[0],
                  index=detected.indices[0][1],
                  ntype='IK',
                  lon=longitudes[detected.indices[0][1]],
                  signal="EW")
        apnd = True
        if len(detected.times.AD) == 1:
            AD = Node(satcat,
                      detected.times.AD[0],
                      index=detected.AD_dex[0],
                      ntype="AD",
                      signal="EW")
            add_node(AD)
        elif len(detected.times.AD) == 0:
            pass
        else:
            for j in range(len(detected.times.AD)):
                ad = Node(satcat,
                          detected.times.AD[j],
                          index=detected.AD_dex[j],
                          ntype="AD",
                          signal="EW")
                ad_next = Node(satcat,
                               detected.times.AD[j+1],
                               index=detected.AD_dex[j+1],
                               ntype="AD",
                               signal="EW") \
                    if j < (len(detected.times.AD)-1) else None
                if (ad.t > starttime+timedelta(hours=detected.indices[0][0]*2)) & (ad.t < IK.t):
                    if apnd & (ad_next is not None):
                        if ((ad_next.t-ad.t) > timedelta(hours=24)):
                            add_node(ad)
                        else:
                            add_node(ad)
                            apnd = False
                    elif apnd & (ad_next is None):
                        add_node(ad)
                    elif (not apnd) & (ad_next is not None):
                        if ((ad_next.t-ad.t) > timedelta(hours=24)):
                            apnd = True
        if detected.indices[0][1] != (len(lon_std)-steps_per_day-1):
            add_node(IK)

    # for each longitudinal shift detection
    for i in range(len(detected.times.IK)-1):
        if toggle:                                                            # if the last ID was not discarded
            # if the time between the current IK & next ID is longer than 48 hours
            if ((starttime+timedelta(hours=detected.indices[i+1][0]*2)-detected.times.IK[i]) > timedelta(hours=36)):
                # keep the current ID
                filtered.times.ID.append(detected.times.ID[i])
                # keep the current IK
                filtered.times.IK.append(detected.times.IK[i])
                ID = Node(satcat,
                          detected.times.ID[i],
                          index=detected.indices[i][0],
                          ntype='ID',
                          lon=longitudes[detected.indices[i][0]],
                          signal="EW")
                add_node(ID)
                IK = Node(satcat,
                          detected.times.IK[i],
                          index=detected.indices[i][1],
                          ntype='IK',
                          lon=longitudes[detected.indices[i][1]],
                          signal="EW")
                apnd = True
                for j in range(len(detected.times.AD)):
                    ad = Node(satcat,
                              detected.times.AD[j],
                              index=detected.AD_dex[j],
                              ntype="AD",
                              signal="EW")
                    ad_next = Node(satcat,
                                   detected.times.AD[j+1],
                                   index=detected.AD_dex[j+1],
                                   ntype="AD",
                                   signal="EW") \
                        if j < (len(detected.times.AD)-1) else None
                    if (ad.t > ID.t) & (ad.t < IK.t):
                        if apnd & (ad_next is not None):
                            if ((ad_next.t-ad.t) > timedelta(hours=24)):
                                add_node(ad)
                            else:
                                add_node(ad)
                                apnd = False
                        elif apnd & (ad_next is None):
                            add_node(ad)
                        elif (not apnd) & (ad_next is not None):
                            if ((ad_next.t-ad.t) > timedelta(hours=24)):
                                apnd = True
                if detected.indices[0][1] != (
                        len(lon_std)-steps_per_day-1):
                    add_node(IK)
                # if the next drift is the last drift
                if i == len(detected.times.IK)-2:
                    # keep the next ID
                    filtered.times.ID.append(
                        starttime+timedelta(hours=detected.indices[i+1][0]*2))
                    ID = Node(satcat,
                              starttime +
                              timedelta(hours=detected.indices[i+1][0]*2),
                              index=detected.indices[i+1][0],
                              ntype='ID',
                              lon=longitudes[detected.indices[i+1][0]],
                              signal="EW")
                    add_node(ID)
                    IK = Node(satcat,
                              detected.times.IK[i+1],
                              index=detected.indices[i+1][1],
                              ntype='IK',
                              lon=longitudes[detected.indices[i+1][1]],
                              signal="EW")
                    apnd = True
                    for j in range(len(detected.times.AD)):
                        ad = Node(satcat,
                                  detected.times.AD[j],
                                  index=detected.AD_dex[j],
                                  ntype="AD",
                                  signal="EW")
                        ad_next = Node(satcat,
                                       detected.times.AD[j+1],
                                       index=detected.AD_dex[j+1],
                                       ntype="AD",
                                       signal="EW") \
                            if j < (len(detected.times.AD)-1) else None
                        if (ad.t > ID.t) & (ad.t < IK.t):
                            if apnd & (ad_next is not None):
                                if ((ad_next.t-ad.t) > timedelta(
                                        hours=24)):
                                    add_node(ad)
                                else:
                                    add_node(ad)
                                    apnd = False
                            elif apnd & (ad_next is None):
                                add_node(ad)
                            elif (not apnd) & (ad_next is not None):
                                if ((ad_next.t-ad.t) > timedelta(
                                        hours=24)):
                                    apnd = True
                    if detected.indices[i][1] != (
                            len(lon_std)-steps_per_day-1):
                        filtered.times.IK.append(
                            detected.times.IK[i+1])      # keep the next IK
                        add_node(IK)
            # if the next ID and the current IK are 48 hours apart or less
            else:
                ID = Node(satcat,
                          detected.times.ID[i],
                          index=detected.indices[i][0],
                          ntype='ID',
                          lon=longitudes[detected.indices[i][0]],
                          signal="EW")                                          # keep the current ID
                add_node(ID)
                AD = Node(satcat,
                          detected.times.IK[i],
                          index=detected.indices[i][1],
                          ntype='AD',
                          lon=longitudes[detected.indices[i][1]],
                          signal="EW")                                          # change the current IK to an AD
                IK = Node(satcat,
                          detected.times.IK[i+1],
                          index=detected.indices[i+1][1],
                          ntype='IK',
                          lon=longitudes[detected.indices[i+1][1]],
                          signal="EW")                                          # exchange the current IK for the next one
                add_node(AD)
                apnd = True
                for j in range(len(detected.times.AD)):
                    ad = Node(satcat,
                              detected.times.AD[j],
                              index=detected.AD_dex[j],
                              ntype="AD",
                              signal="EW")
                    ad_next = Node(satcat,
                                   detected.times.AD[j+1],
                                   index=detected.AD_dex[j+1],
                                   ntype="AD",
                                   signal="EW") \
                        if j < (len(detected.times.AD)-1) else None
                    if (ad.t > ID.t) & (ad.t < IK.t):
                        if apnd & (ad_next is not None):
                            if ((ad_next.t-ad.t) > timedelta(hours=24)):
                                add_node(ad)
                            else:
                                add_node(ad)
                                apnd = False
                        elif apnd & (ad_next is None):
                            add_node(ad)
                        elif (not apnd) & (ad_next is not None):
                            if ((ad_next.t-ad.t) > timedelta(hours=24)):
                                apnd = True
                if detected.indices[0][1] != (
                        len(lon_std)-steps_per_day-1):
                    add_node(IK)
                filtered.times.ID.append(detected.times.ID[i])
                filtered.times.AD.append(detected.times.IK[i])
                filtered.times.IK.append(detected.times.IK[i+1])
                # skip the redundant drift
                toggle = False
        else:
            toggle = True
    add_node(es)
    return nodes, filtered


def detect_sn_pol_nodes(data, nodes, filtered, satcat, starttime, steps_per_day):

    inclinations = data["Inclination (deg)"]
    ssNS = Node(
        satcat=satcat,
        t=starttime,
        index=0,
        ntype="SS",
        signal="NS")
    for j in range(len(filtered.indices)):
        indices = filtered.indices[j]
        first = True if indices[0] == 0 else False
        times = []
        dexs = []
        inc = inclinations[indices[0]:indices[1]].to_numpy()
        t = np.arange(indices[0], indices[1])*2
        rate = (steps_per_day/(indices[1]-indices[0])
                )*(np.max(inc)-np.min(inc))
        XIPS_inc_per_day = 0.0005  # 0.035/30
        if (rate < XIPS_inc_per_day) and (indices[0] < steps_per_day) and (indices[1] > steps_per_day):
            if filtered.modes.end[j]:
                nodes.append(Node(
                    satcat=satcat,
                    t=starttime+timedelta(hours=indices[1]*2),
                    index=indices[1],
                    ntype="ID",
                    signal="NS",
                    mtype="NK"
                ))
            ssNS.mtype = filtered.modes.SK[j]
        elif (rate < XIPS_inc_per_day):
            nodes.append(Node(
                satcat=satcat,
                t=starttime+timedelta(hours=indices[1]*2),  # times[0],
                index=indices[1],  # dexs[0],
                # t=times[0],
                # index=dexs[0],
                ntype="IK",
                signal="NS",
                mtype=filtered.modes.SK[j]
            ))
            if filtered.modes.end[j]:
                nodes.append(Node(
                    satcat=satcat,
                    t=starttime+timedelta(hours=indices[1]*2),
                    index=indices[1],
                    ntype="ID",
                    signal="NS",
                    mtype="NK"
                ))
        else:
            dt = [0.0]
            for i in range(len(inc)-1):
                dt.append((inc[i+1]-inc[i])/(2*60*60))
            prev = 1.0
            for i in range(len(dt)-1):
                if np.abs(dt[i]) > 5.5e-7:
                    times.append(starttime+timedelta(hours=float(t[i])))
                    dexs.append(i+indices[0])
                    if (np.abs(np.mean(inc[0:i])-np.mean(inc[i:len(inc)]))/np.std(inc[0:i]))/prev < 1.0:
                        if first and len(times) == 2:
                            ssNS.mtype = filtered.modes.SK[0]
                            first = False
                    elif len(times) == 2:
                        first = False
                    prev = np.abs(
                        np.mean(inc[0:i])-np.mean(inc[i:len(inc)]))/np.std(inc[0:i])

            if len(times) > 0:
                nodes.append(Node(
                    satcat=satcat,
                    t=times[0],
                    index=dexs[0],
                    ntype="IK",
                    signal="NS",
                    mtype=filtered.modes.SK[j]
                ))
                if filtered.modes.end[j]:
                    nodes.append(Node(
                        satcat=satcat,
                        t=starttime+timedelta(hours=indices[1]*2),
                        index=indices[1],
                        ntype="ID",
                        signal="NS",
                        mtype="NK"
                    ))
            elif filtered.indices[0][0] == 0:
                ssNS.mtype = filtered.modes.SK[0]
            else:
                ssNS.mtype = "NK"
    nodes.append(ssNS)
    nodes.sort(key=lambda x: x.t)
    return nodes


def data_post_processing(nodes, starttime):
    # Convert timestamp back into timeindex and format the output to the correct format in a pandas dataframe
    ObjectID_list = []
    TimeIndex_list = []
    Direction_list = []
    Node_list = []
    Type_list = []
    for i in range(len(nodes)):
        ObjectID_list.append(nodes[i].satcat)
        TimeIndex_list.append(
            int(((nodes[i].t-starttime).days*24+(nodes[i].t-starttime).seconds/3600)/2))
        Direction_list.append(nodes[i].signal)
        Node_list.append(nodes[i].type)
        Type_list.append(nodes[i].mtype)

    # Initialize data of lists.
    data = {'ObjectID': ObjectID_list,
            'TimeIndex': TimeIndex_list,
            'Direction': Direction_list,
            'Node': Node_list,
            'Type': Type_list}

    # Create the pandas DataFrame
    prediction = pd.DataFrame(data)

    return prediction


def data_to_add(data_df, ground_truth_object):
    ground_truth_object.Type = ground_truth_object.Type.fillna('NK')
    ground_truth_EW = ground_truth_object[ground_truth_object['Direction'] == 'EW'].copy(
    )
    ground_truth_NS = ground_truth_object[ground_truth_object['Direction'] == 'NS'].copy(
    )

    # Create 'EW' and 'NS' labels and fill 'unknown' values
    ground_truth_EW['EW_baseline_heuristic'] = ground_truth_EW['Node'] + \
        '-' + ground_truth_EW['Type']
    ground_truth_NS['NS_baseline_heuristic'] = ground_truth_NS['Node'] + \
        '-' + ground_truth_NS['Type']
    ground_truth_EW.drop(
        ['Node', 'Type', 'Direction'], axis=1, inplace=True)
    ground_truth_NS.drop(
        ['Node', 'Type', 'Direction'], axis=1, inplace=True)
    merged_df = pd.merge(data_df,
                         ground_truth_EW.sort_values('TimeIndex'),
                         on=['TimeIndex', 'ObjectID'],
                         how='left')
    merged_df = pd.merge_ordered(merged_df,
                                 ground_truth_NS.sort_values(
                                     'TimeIndex'),
                                 on=['TimeIndex', 'ObjectID'],
                                 how='left')

    # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
    merged_df['EW_baseline_heuristic_ffill'] = merged_df['EW_baseline_heuristic'].ffill()
    merged_df['NS_baseline_heuristic_ffill'] = merged_df['NS_baseline_heuristic'].ffill()

    merged_df['EW_baseline_heuristic'].fillna('Nothing', inplace=True)
    merged_df['NS_baseline_heuristic'].fillna('Nothing', inplace=True)

    return merged_df


def add_baseline_heuristic(data, objectId, starttime, endtime, feature_cols):
    detected, ssEW, lon_std = detect_ew_pol_nodes(
        data.loc[data.ObjectID == objectId][feature_cols], objectId, starttime, endtime)
    nodes, filtered = filter_and_merge_nodes(
        data.loc[data.ObjectID == objectId][feature_cols], detected, ssEW, lon_std,   objectId, starttime, endtime, 12)
    nodes = detect_sn_pol_nodes(
        data, nodes, filtered, objectId, starttime, 12)
    merged_df = data_to_add(
        data.loc[data.ObjectID == objectId], data_post_processing(nodes, starttime))
    return merged_df


