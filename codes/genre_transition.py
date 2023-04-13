def feature_extraction(file_path, file_name):
    
    import librosa
    import numpy as np
    new_datas = []

    y, sr = librosa.load(file_path)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    chroma_stft_var = np.var(librosa.feature.chroma_stft(y=y, sr=sr))
    rmse = np.mean(librosa.feature.rms(y=y))
    rmse_var = np.var(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_cent_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spec_bw_var = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rolloff_var = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    zcr_var = np.var(librosa.feature.zero_crossing_rate(y))
    new_datas.append({
    "path":file_path,
    "filename":file_name,
    "chroma_stft":chroma_stft,
    "rmse":rmse,
    "spec_cent":spec_cent,
    "spec_bw":spec_bw,
    "rolloff":rolloff,
    "zcr":zcr,
    "chroma_stft_var":chroma_stft_var,
    "rmse_var":rmse_var,
    "spec_cent_var":spec_cent_var,
    "spec_bw_var":spec_bw_var,
    "rolloff_var":rolloff_var,
    "zcr_var":zcr_var
    })

    return new_datas

def mfcc_extraction(file_path, mfcc_n, file_name):

    import librosa
    import numpy as np

    head_mfcc = []
    for i in range(1, mfcc_n+1):
        head_mfcc.append(f'mfcc{i}')
    head_mfcc_var = []
    for i in range(1, mfcc_n+1):
        head_mfcc_var.append(f'mfcc_var{i}')

    new_datas_mfcc = []
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_n)
    mfccs = {}
    for e in range(0,len(mfcc)):
        mfccs[head_mfcc[e]]=np.mean(mfcc[e])
        mfccs[head_mfcc_var[e]]=np.var(mfcc[e])
    datas_tmp={
    "path":file_path,
    "filename":file_name
    }
    datas_tmp = dict(datas_tmp, **mfccs)
    new_datas_mfcc.append(datas_tmp)

    return new_datas_mfcc

def feature_extraction_by_folder(folder_path):
    
    import librosa
    import numpy as np
    import os

    m_files = []
    directories = []
    for i in os.walk(folder_path):
        for j in i:
            if isinstance(j, list) and j!=[]:
                m_files.append(j)
            elif j:
                directories.append(j)
    m_files.remove(['.DS_Store'])
    wave_files = dict(zip(directories[1:], m_files[1:]))
    
    Y = []
    datas = []
    for i, i1 in wave_files.items():
        for j in i1:
            data = os.path.join(i, j)
            y, sr = librosa.load(data)
            chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            chroma_stft_var = np.var(librosa.feature.chroma_stft(y=y, sr=sr))
            rmse = np.mean(librosa.feature.rms(y=y))
            rmse_var = np.var(librosa.feature.rms(y=y))
            spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spec_cent_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
            spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spec_bw_var = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            rolloff_var = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            zcr_var = np.var(librosa.feature.zero_crossing_rate(y))
            datas.append({
            "path":data,
            "filename":j,
            "chroma_stft":chroma_stft,
            "rmse":rmse,
            "spec_cent":spec_cent,
            "spec_bw":spec_bw,
            "rolloff":rolloff,
            "zcr":zcr,
            "chroma_stft_var":chroma_stft_var,
            "rmse_var":rmse_var,
            "spec_cent_var":spec_cent_var,
            "spec_bw_var":spec_bw_var,
            "rolloff_var":rolloff_var,
            "zcr_var":zcr_var
            })
            Y.append(i)
    return datas, Y


def mfcc_extraction_by_folder(folder_path, mfcc_n):
    
    import librosa
    import numpy as np
    import os
    
    m_files = []
    directories = []
    for i in os.walk(folder_path):
        for j in i:
            if isinstance(j, list) and j!=[]:
                m_files.append(j)
            elif j:
                directories.append(j)
    wave_files = dict(zip(directories[1:], m_files[1:]))


    head_mfcc = []
    for i in range(1, mfcc_n+1):
        head_mfcc.append(f'mfcc{i}')
    head_mfcc_var = []
    for i in range(1, mfcc_n+1):
        head_mfcc_var.append(f'mfcc_var{i}')
    datas_mfcc = []
    Y = []
    for i, i1 in wave_files.items():
        for j in i1:
            data = os.path.join(i, j)
            y, sr = librosa.load(data)
            mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=mfcc_n)
            mfccs = {}
            for e in range(0,len(mfcc)):
                mfccs[head_mfcc[e]]=np.mean(mfcc[e])
                mfccs[head_mfcc_var[e]]=np.var(mfcc[e])
            datas_tmp={
            "path":data,
            "filename":j
            }
            datas_tmp = dict(datas_tmp, **mfccs)
            datas_mfcc.append(datas_tmp)
            Y.append(i)
    return datas_mfcc, Y