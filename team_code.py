#!/usr/bin/env python
import pandas as pd
# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################
from sklearn.impute import KNNImputer
from helper_code import *
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import numpy as np, os, sys
from CCA_NEW import *
from sklearn.decomposition import PCA
import mne
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from sklearn.linear_model import LogisticRegression
from feature_selections_relifF import *


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.

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



def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)#病人id，应该是列表存储
    num_patients = len(patient_ids)#这里的

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)#创建文件目录

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)
        #用于批处理的问你件名加载的，patient_metadata是txt文件，记录个人信息，而recording_metadata记录着tsv文件（一个比较复杂的评价标准）

        # Extract features.提取特征，可以针对这里进行修改
        current_features = get_features(patient_metadata, recording_metadata, recording_data)
        features.append(current_features)

        # Extract labels.提取模型
        current_outcome = get_outcome(patient_metadata)#从数据中提取outcome,好是0，坏是1
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)#同理获得cpc
        cpcs.append(current_cpc)

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')


    # Impute any missing features; use the mean value by default.
    #缺失值填充，使用了平均值来填充
    imputer = SimpleImputer().fit(features)

    # Train the models.
    # Define parameters for random forest classifier and regressor.
    n_estimators = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state = 789  # Random state; set for reproducibility.
    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)
    # 处理数据
    include_file = np.column_stack((features, outcomes))
    CCA_model = fit(include_file)
    features_rfe = rfe(features, outcomes)
    relief = RelifF(10, 0.1, 5)
    features_relief = relief.fit_transform(features, outcomes.ravel())

    outcome_model = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,
                                           random_state=random_state).fit(features, outcomes.ravel())

    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model, CCA_model, features)

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
    CCA_model = models['CCA_model']
    full_features = models['full_features']

    # Load data.
    #patient_metadata是'Patient: ICARE_0284、Age: 53Sex: MaleROSC: nan，OHCA: TrueVFib: TrueTM: 33Outcome: GoodCPC: 1此内容
    # recording_metadata是这样一个文本值，这里其实是读的tsv文件，他官方的代码就只选用前12个小时的
    # Hour	Time	Quality	Record
    # 01	nan	nan	nan
    # 02	nan	nan	nan
    # 03	nan	nan	nan
    # 04	nan	nan	nan
    # 05	04:25	1.000	ICARE_0284_05
    # 06	05:55	1.000	ICARE_0284_06
    # 07	06:50	1.000	ICARE_0284_07
    # 08	07:55	1.000	ICARE_0284_08
    # 09	08:55	0.983	ICARE_0284_09
    # 10	09:55	1.000	ICARE_0284_10
    # 11	10:30	1.000	ICARE_0284_11
    # 12	11:00	0.817	ICARE_0284_12
    #recording是前12个的内容，其中除非是nan，否则都会是18*30000的array
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    # Extract features.
    features = get_features(patient_metadata, recording_metadata, recording_data)
    features = features.reshape(1, -1)
    #outcomes = get_outcome(patient_metadata)
    # Impute missing data.
    features = imputer.transform(features)
    # Apply models to features.
    outcome = bagging_outcome(outcome_model, CCA_model, features, full_features)#集成所有的outcome结果


    #print("outcome预测结果为：", outcome)
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = bagging_cpc(cpc_model, features)
    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

def bagging_outcome(outcome_model, CCA_model, features, full_features):
    outcome_list = []
    #features_rfe = rfe(features, outcomes)#rfe处理数据集
    #features_relief = reliefF(features, outcomes)#relief处理数据集
    outcomes = list()
    outcome1 = outcome_model.predict(features)[0]
    outcome2 = CCA_test(full_features, features, CCA_model)

    #print("随机森林结果",outcome1)
    #print("CCA结果",outcome2)
    outcome_list.append(outcome1)
    outcome_list.append(outcome2)
    x = 0
    y = 0
    for i in range(len(outcome_list)):
        if outcome_list[i] == 0 :
            x = x+1
        else :
            y = y+1
    if x >= y :
        outcome = 0
    else :
        outcome = random.choice([0, 1])
    return  outcome

def bagging_cpc(cpc_model, features):
    cpc1 = cpc_model.predict(features)[0]

    return  cpc1




# ########################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
###################################### ##########################################


# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model, CCA_model, features):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model,'CCA_model': CCA_model,'full_features': features}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
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

    # Combine the patient features,【年龄，性别等信息】
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])

    # Extract features from the recording data and metadata.提取特征
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            available_signal_data.append(signal_data)
#电路图数据处理
    if len(available_signal_data) > 0:
        #因为第一个0284这个人取12个h，根据头文件只有8个小时能用，所以这里的数据就是18x30000x8 = 18x240000
        available_signal_data = np.hstack(available_signal_data)
        available_signal_data_pca = PCA(n_components= 0.95)#111111111111111111111111111111111111111111111111111111111111111111111111111
        available_signal_data = available_signal_data_pca.fit_transform(available_signal_data)

        available_signal_data_csv = pd.DataFrame(available_signal_data)
        #available_signal_data_csv.to_csv("C:/Users/27689/Desktop/PCA_signal_data.csv", index=0)

        #计算平均值，方差，18列
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std  = np.nanstd(available_signal_data, axis=1)
    #大致理解为平均值和方差值
    else:
        signal_mean = float('nan') * np.ones(num_channels)
        signal_std  = float('nan') * np.ones(num_channels)

    # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of the most
    # recent recording.
    index = None
    for i in reversed(range(num_recordings)):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            index = i
            break

    if index is not None:
        signal_data, sampling_frequency, signal_channels = recording_data[index]
        signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.

        signal_data_csv = pd.DataFrame(signal_data)
        #signal_data_csv.to_csv("C:/Users/27689/Desktop/signal.csv", index=0)

        delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_pca = PCA(n_components=0.5)
        delta_psd = delta_psd_pca.fit_transform(delta_psd)
        theta_psd_pca = PCA(n_components=0.5)
        theta_psd = delta_psd_pca.fit_transform(delta_psd)
        alpha_psd_pca = PCA(n_components=0.5)
        alpha_psd = delta_psd_pca.fit_transform(delta_psd)
        beta_psd_pca = PCA(n_components=0.5)
        beta_psd = delta_psd_pca.fit_transform(delta_psd)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

        quality_score = get_quality_scores(recording_metadata)[index]
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
        quality_score = float('nan')
    #依次的长度为18*6 = 108 +1 =109
    recording_features = np.hstack((signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))
    #signal_mean（18行的所有平均值），signal_std也是18个，使用的全部都是平均值，18x6
    # Combine the features from the patient metadata and the recording data and metadata.np.vstack():在竖直方向上堆叠,np.hstack():在水平方向上平铺
    #拼接array，patient_feature的长度为8个，recording的长度为109个，共计117个。
    features = np.hstack((patient_features, recording_features))

    return features
