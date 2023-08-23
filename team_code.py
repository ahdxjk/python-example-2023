#!/usr/bin/env python
# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from sklearn.impute import KNNImputer
from helper_code import *
import tsfresh
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters, settings
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import numpy as np, os, sys
from CCA_NEW import *
from sklearn.decomposition import PCA
import mne
from sklearn import ensemble
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from sklearn.linear_model import LogisticRegression
from feature_selections_relifF import *
import pandas as pd
import  pywt
from scipy import signal

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

        if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        current_features = get_features(data_folder, patient_ids[i])
        features.append(current_features)

        # Extract labels.
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 789  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)


    # Train the models.
    features = imputer.transform(features)
    #pycaret
    #csv1 = np.hstack((features, outcomes))
    #csv2 = np.hstack((features,cpcs))
    #csv1 = pd.DataFrame(csv1)
    #csv2 = pd.DataFrame(csv2)
    #csv1.to_csv("/home/coding/outcome")
    #csv2.to_csv("/home/coding/cpc")


    #PCA 和 RELIEF 特征选择
    #pca = PCA(n_components=400)
    #features = pca.fit_transform(features)
    #print(features.shape)


    #CCA model
    include_file = np.column_stack((features, outcomes))
    include_file_cpc = np.column_stack((features, cpcs))
    CCA_model_outcomes = fit(include_file)
    CCA_model_cpcs = fit(include_file_cpc)


    #rf
    outcome_model = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,
                                           random_state=random_state).fit(features, outcomes.ravel())

    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, cpcs.ravel())

    #gbc
    gbc_outcome_model = GradientBoostingClassifier(n_estimators=123, learning_rate=1.0, max_depth = 1, random_state = 789).fit(features, outcomes.ravel())
    gbc_cpc_model = GradientBoostingClassifier(n_estimators=123, learning_rate=1.0, max_depth=1, random_state=789).fit(features, cpcs.ravel())

    #ada

    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model, CCA_model_outcomes, CCA_model_cpcs, features, gbc_outcome_model , gbc_cpc_model)

    if verbose >= 1:
        print('Done.')


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']
    CCA_model_outcomes = models['CCA_model_outcomes']
    CCA_model_cpcs = models['CCA_model_cpcs']
    full_features = models['full_features']
    gbc_outcome_model = models['gbc_outcome_model']
    gbc_cpc_model = models['gbc_cpc_model']



    # Extract features.
    features = get_features(data_folder, patient_id)
    features = features.reshape(1, -1)
    #print("在运行阶段提取到的特征1为",features.shape)
    # Impute missing data.
    #print(features.shape)
    #print(full_features.shape)
    if features.shape[1] != full_features.shape[1] :
        x = full_features.shape[1] - features.shape[1]
        for i in range(0, x):
            connect = np.arange(0, features.shape[0])
            features = np.column_stack((features, connect))

    features = imputer.transform(features)
    #print("在运行阶段提取到的特征2为",features.shape)

    # Apply models to features.1
    outcome = bagging_outcome(outcome_model, CCA_model_outcomes, features, full_features, gbc_outcome_model)#集成所有的outcome结果


    #print("outcome预测结果为：", outcome)
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = bagging_cpc(cpc_model, CCA_model_cpcs, features, full_features, gbc_cpc_model)
    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)
    #print("result", outcome ,end= " ")
    #print(cpc)
    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model, CCA_model_outcomes, CCA_model_cpcs, features, gbc_outcome_model , gbc_cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model,'CCA_model_outcomes': CCA_model_outcomes,
         'CCA_model_cpcs': CCA_model_cpcs,'full_features': features, 'gbc_outcome_model' : gbc_outcome_model, 'gbc_cpc_model' : gbc_cpc_model }
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data

    return data, resampling_frequency

# Extract features.
def get_features(data_folder, patient_id):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)

    # Extract EEG features.
    # eeg_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3','F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    eeg_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
    group = 'EEG'


    if num_recordings > 0:
        recording_id = recording_ids[-1]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            if all(channel in channels for channel in eeg_channels):
                data, channels = reduce_channels(data, channels, eeg_channels)
                data
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                data = np.array([data[0, :] - data[2, :], data[2, :] - data[6, :], data[6,:] - data[10,:],data[ 10,:] - data[14,:]
                                ,data[1,:] - data[3,:], data[3,:] - data[7,:], data[7,:] - data[11,:], data[11,:] - data[15,:]
                                ,data[0,:] - data[4,:], data[4,:] - data[8,:], data[8,:] - data[12], data[12,:] - data[14 ,:]
                                ,data[1,:] - data[5,:], data[5,:] - data[9,:], data[9,:] - data[13], data[13,:] - data[15 ,:]
                                ,data[16,:] - data[17,:], data[17, :] - data[18,:]])
                #data = data[ : , :320000]
                eeg_features = get_eeg_features(data, sampling_frequency).flatten()
            else:
                eeg_features = float('nan') * np.ones(8) # 2 bipolar channels * 4 features / channel
        else:
            eeg_features = float('nan') * np.ones(8) # 2 bipolar channels * 4 features / channel
    else:
        eeg_features = float('nan') * np.ones(8) # 2 bipolar channels * 4 features / channel
    print('eeeeeeg', eeg_features.shape)
    # Extract ECG features.
    ecg_channels = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
    group = 'ECG'

    if num_recordings > 0:
        recording_id = recording_ids[0]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            data, channels = reduce_channels(data, channels, ecg_channels)
            data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
            features = get_ecg_features(data)
            ecg_features = expand_channels(features, channels, ecg_channels).flatten()
        else:
            ecg_features = float('nan') * np.ones(50) # 5 channels * 2 features / channel
    else:
        ecg_features = float('nan') * np.ones(50) # 5 channels * 2 features / channel
    print('ecgggggg', ecg_features.shape)
    # Extract features.
    hhh = np.hstack((patient_features, eeg_features, ecg_features))
    print('feature',hhh.shape)
    return np.hstack((patient_features, eeg_features, ecg_features))

# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.hstack((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))
    #print("patient_feature", features.shape)
    return features

# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    #abc = pd.DataFrame(data)
    #abc.to_csv("/home/coding/sample_data")
    #exit(1)

    if num_samples > 0:
        # 首先是时域特征，均值，方差，偏斜度等等特征
        signal_mean = np.nanmean(data, axis=1)  # 均值
        signal_std = np.nanstd(data, axis=1)  # 方差
        signal_max = np.nanmax(data, axis=1)  # 最大值
        signal_min = np.nanmin(data, axis=1)  # 最小值
        signal_var = np.nanvar(data, axis=1)  # 标准差
        signal_sc = []
        for i in range(0, len(data) - 1):
            data_single = data[i][:]
            data_single_mean = np.mean(data_single)
            data_single_std = np.std(data_single, ddof=1)
            signal_sc.append(np.mean(((data_single - data_single_mean) / data_single_std) ** 3))
        signal_sc = np.array(signal_sc)  # 计算偏斜度

        # 大致理解为平均值和方差值
        signal_data_get_feature = expand_feature(data)
        signal_data_get_feature = signal_data_get_feature.ravel()
        #print("expand", signal_data_get_feature.shape)
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        #平均值和总和方差
        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

        delta_psd_sum = np.nansum(delta_psd, axis=1)
        theta_psd_sum = np.nansum(theta_psd, axis=1)
        alpha_psd_sum = np.nansum(alpha_psd, axis=1)
        beta_psd_sum = np.nansum(beta_psd, axis=1)

        delta_psd_std = np.nanstd(delta_psd, axis=1)
        theta_psd_std = np.nanstd(theta_psd, axis=1)
        alpha_psd_std = np.nanstd(alpha_psd, axis=1)
        beta_psd_std = np.nanstd(beta_psd, axis=1)


        #计算能量占比
        welch_full  = np.hstack((delta_psd, theta_psd, alpha_psd, beta_psd))
        delta_occupy  = np.array(np.nansum(delta_psd) / np.nansum(welch_full))
        theta_occupy = np.array(np.nansum(theta_psd) / np.nansum(welch_full))
        alpha_occupy = np.array(np.nansum(alpha_psd) / np.nansum(welch_full))
        beta_occupy = np.array(np.nansum(beta_psd) / np.nansum(welch_full))
        #print(delta_occupy)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
        delta_psd_sum = theta_psd_sum = alpha_psd_sum = beta_psd_sum = float('nan') * np.ones(num_channels)

    #小波包分解
    WaveletPacket_feature = dwt(data,5)
    features = np.hstack((signal_mean, signal_std, signal_max, signal_min, signal_sc,
                          signal_var,delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean,
                          delta_psd_sum, theta_psd_sum, alpha_psd_sum, beta_psd_sum,
                          delta_psd_std, theta_psd_std, alpha_psd_std, beta_psd_std,
                          delta_occupy, theta_occupy, alpha_occupy, beta_occupy,
                          WaveletPacket_feature, signal_data_get_feature)).T
    #print("eeg features", features.shape)
    return features

# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)
    if num_samples > 1024:
        #时域特征平均值，方差
        mean = np.mean(data, axis=1)
        ecg_dwt_list = []
        psd_list =  []
        std  = np.std(data, axis=1)
        print(data.shape)
        #时频特征
        if data.shape[0] == 2:
            for i in range(0, data.shape[0]):
                data1 = data[i, :]
                ecg_dwt_feature = ecg_dwt(data1, 5)
                ecg_dwt_list.append(ecg_dwt_feature)
                psd_feature = ecg_psd(data)
                psd_list.append(psd_feature)

        else :
            psd_feature = ecg_psd(data)
            psd_list.append(psd_feature)
            ecg_dwt_feature = ecg_dwt(data, 5)
            ecg_dwt_list.append(ecg_dwt_feature)

        ecg_dwt_array = np.hstack(ecg_dwt_list)
        psd_array = np.hstack(psd_list)
        features = np.vstack((mean, std, ecg_dwt_array, psd_array))
        features = features.T

    else:
        features = float('nan') * np.ones(50)

    return features

def rfe(features, outcomes):
    estimator = LogisticRegression()
    selector = RFE(estimator, n_features_to_select=0.5, step=1)
    #selector是经过rfe筛选过后的特征
    selector = selector.fit_transform(features, outcomes)
    return selector

def reliefF(features, outcomes):
    features = np.array(features)
    outcomes = np.ravel(outcomes)
    reliefF_classify = RelifF(10, 0.1, 5)
    relief_features = reliefF_classify.fit_transform(features, outcomes)
    return relief_features


def bagging_outcome(outcome_model, CCA_model, features, full_features, gbc_outcome_model):
    outcome_list = []
    outcomes = list()
    outcome1 = outcome_model.predict(features)[0]
    outcome2 = CCA_test(full_features, features, CCA_model)
    outcome3 = gbc_outcome_model.predict(features)
    #print("随机森林结果",outcome1)
    #print("CCA结果",outcome2)
    #print("pbc结果", outcome3)
    outcome_list.append(outcome1)
    outcome_list.append(outcome2)
    outcome_list.append(outcome3)
    #集成硬投票
    x = 0
    y = 0
    for i in range(len(outcome_list)):
        if outcome_list[i] == 0 :
            x = x+1
        else :
            y = y+1
    if x > y :
        outcome = 0
    else :
        outcome = 1

    return  outcome

def bagging_cpc(cpc_model, CCA_model_cpcs, features, full_features, gbc_cpc_model):
    cpc1 = cpc_model.predict(features)[0]
    cpc2 = CCA_test(full_features, features, CCA_model_cpcs)
    cpc3 = gbc_cpc_model.predict(features)[0]
    #print("随机森林cpc",cpc1)
    #print("CCA结果cpc",cpc2)
    #print("pbc结果cpc", cpc3)
    cpc = (cpc2 + cpc3)/2
    return  cpc

def expand_feature(data):
    data_df = pd.DataFrame(data)
    data_t = data_df.T
    data_t.insert(loc=0, column= 'time', value = range(len(data_t)))
    data_t.insert(loc=0, column='id', value = 1)
    data_t['time'] = range(len(data_t))
    extracted_features = extract_features(data_t, column_id="id", default_fc_parameters=settings.MinimalFCParameters() )
    extracted_features_t = extracted_features.T
    #print(extracted_features_t)
    return extracted_features_t.values

def dwt(signal, n):
    wp = pywt.WaveletPacket(data=signal, wavelet='db3', mode='symmetric', maxlevel=n)
    re = []  # 第n层所有节点的分解系数
    for i in [node.path for node in wp.get_level(n, 'freq')]:
       re.append(wp[i].data)
    # 第n层能量特征
    energy = []
    occupy_features = []
    for i in re:
       energy.append(pow(np.linalg.norm(i, ord=None), 2))
    sum_num = sum(energy)
    #计算每每个频段在总体占比，平均值，方差
    mean = np.mean(re)
    std = np.std(re)
    for i in range(len(energy)):
        occupy_features.append(energy[i] / sum_num)


    #小波分解计算熵特征等
    #coeffs = pywt.wavedec(data = signal, wavelet='db3', level=5)
    energy_cout = np.square(re)
    energy_ratio = energy_cout / np.sum(energy_cout)
    entropy = -np.sum(energy_ratio * np.log2(energy_ratio))
    entropy = np.array(entropy)
    #print("能量熵:", entropy)

    # bagging_feature
    energy = np.array(energy)
    occupy_features = np.array(occupy_features)
    features = np.hstack((energy, occupy_features, entropy, mean, std))
    #print("小波特征", features.shape)

    return  features


def ecg_dwt(signal, n):
    wp = pywt.WaveletPacket(data=signal, wavelet='db3', mode='symmetric', maxlevel=n)
    re = []  # 第n层所有节点的分解系数
    for i in [node.path for node in wp.get_level(n, 'freq')]:
        re.append(wp[i].data)
    # 第n层能量特征
    energy = []
    occupy_features = []
    for i in re:
        energy.append(pow(np.linalg.norm(i, ord=None), 2))
    sum_num = sum(energy)
    # 计算每每个频段在总体占比，平均值，方差
    mean = np.mean(re)
    #print('1',mean)
    std = np.std(re)
    occupy_features.append(energy[0] / sum_num)
    #print('2',occupy_features)

    # 小波分解计算熵特征等
    # coeffs = pywt.wavedec(data = signal, wavelet='db3', level=5)
    energy_cout = np.square(re)
    energy_ratio = energy_cout / np.sum(energy_cout)
    entropy = -np.sum(energy_ratio * np.log2(energy_ratio))
    entropy = np.array(entropy)
    #print("能量熵:", entropy)

    features = np.vstack(( occupy_features, entropy, mean, std))
    #print("小波特征", features.shape)

    return features


def ecg_psd(ecg_signal):
    fs = 100
    frequencies, psd = signal.welch(ecg_signal, fs=fs, window='hann', nperseg=1024, noverlap=512)
    psd_mean = np.mean(psd)
    psd_max = np.max(psd)
    psd_std = np.std(psd)
    psd_sum = np.sum(psd)

    psd_feature = np.vstack((psd_mean, psd_std, psd_max, psd_sum))
    return psd_feature

