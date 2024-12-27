import pickle
import librosa
import numpy as np
def feature_extraction(file_name):
    data,sample_rate=librosa.load(file_name)
    mfccs_features=librosa.feature.mfcc(y=data,sr=sample_rate,n_mfcc=40)
    mfccs_scaled_features=np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

with open('decision_tree.pkl','rb') as model_file:
    loaded_model=pickle.load(model_file)
data_test=feature_extraction('one_test.wav')
data_test = data_test.reshape(1, -1)
label=loaded_model.predict(data_test)
print(label)