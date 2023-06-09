def feature_extraction(file_path, file_name, n):
    
    import librosa
    import numpy as np
    new_datas = []

    y, sr = librosa.load(file_path)
    dur = librosa.get_duration(y=y, sr=sr)
    start = 0

    for k in range(0, int(dur//n)):
        new_y, new_sr = librosa.load(file_path,offset=start,duration=n)

        chroma_stft = np.mean(librosa.feature.chroma_stft(y=new_y, sr=new_sr))
        chroma_stft_var = np.var(librosa.feature.chroma_stft(y=new_y, sr=new_sr))
        rmse = np.mean(librosa.feature.rms(y=new_y))
        rmse_var = np.var(librosa.feature.rms(y=new_y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=new_y, sr=new_sr))
        spec_cent_var = np.var(librosa.feature.spectral_centroid(y=new_y, sr=new_sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=new_y, sr=new_sr))
        spec_bw_var = np.var(librosa.feature.spectral_bandwidth(y=new_y, sr=new_sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=new_y, sr=new_sr))
        rolloff_var = np.var(librosa.feature.spectral_rolloff(y=new_y, sr=new_sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(new_y))
        zcr_var = np.var(librosa.feature.zero_crossing_rate(new_y))
        harmony_y, puccsive_y = librosa.effects.hpss(new_y)
        harmony_mean = np.mean(librosa.effects.harmonic(harmony_y))
        harmony_var = np.var(librosa.effects.harmonic(harmony_y))
        perceptr_mean = np.mean(puccsive_y)
        perceptr_var = np.var(puccsive_y)
    
        
        new_datas.append({
        "path":file_path,
        "split":f"{file_name}_{k}",
        "filename":file_name,
        "chroma_stft_mean":chroma_stft,
        "rms_mean":rmse,
        "spectral_centroid_mean":spec_cent,
        "spectral_bandwidth_mean":spec_bw,
        "rolloff_mean":rolloff,
        "zero_crossing_rate_mean":zcr,
        "chroma_stft_var":chroma_stft_var,
        "rms_var":rmse_var,
        "spectral_centroid_var":spec_cent_var,
        "spectral_bandwidth_var":spec_bw_var,
        "rolloff_var":rolloff_var,
        "zero_crossing_rate_var":zcr_var,
        "harmony_mean": harmony_mean,
        "harmony_var": harmony_var,
        "perceptr_mean": perceptr_mean,
        "perceptr_var" : perceptr_var
        })
        start += n

    return new_datas

def mfcc_extraction(file_path, mfcc_n, file_name, n):

    import librosa
    import numpy as np

    head_mfcc = []
    for i in range(1, mfcc_n+1):
        head_mfcc.append(f'mfcc{i}_mean')
    head_mfcc_var = []
    for i in range(1, mfcc_n+1):
        head_mfcc_var.append(f'mfcc{i}_var')

    y, sr = librosa.load(file_path)
    dur = librosa.get_duration(y=y, sr=sr)
    new_datas_mfcc = []
    start = 0
    for k in range(0, int(dur//n)):
        new_y, new_sr = librosa.load(file_path,offset=start, duration=n)
        mfcc = librosa.feature.mfcc(y=new_y, sr=new_sr, n_mfcc=mfcc_n)
        mfccs = {}
        for e in range(0,len(mfcc)):
            mfccs[head_mfcc[e]]=np.mean(mfcc[e])
            mfccs[head_mfcc_var[e]]=np.var(mfcc[e])
        datas_tmp={
            "path":file_path,
            "split":f"{file_name}_{k}",
            "filename":file_name
            }
        datas_tmp = dict(datas_tmp, **mfccs)
        new_datas_mfcc.append(datas_tmp)
        start += n

    return new_datas_mfcc

def feature_extraction_by_folder(folder_path,n):
    
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
            dur = librosa.get_duration(y=y, sr=sr)
            start = 0

            for k in range(0, int(dur//n)):
                new_y, new_sr = librosa.load(data,offset=start,duration=n)

                chroma_stft = np.mean(librosa.feature.chroma_stft(y=new_y, sr=new_sr))
                chroma_stft_var = np.var(librosa.feature.chroma_stft(y=new_y, sr=new_sr))
                rmse = np.mean(librosa.feature.rms(y=new_y))
                rmse_var = np.var(librosa.feature.rms(y=new_y))
                spec_cent = np.mean(librosa.feature.spectral_centroid(y=new_y, sr=new_sr))
                spec_cent_var = np.var(librosa.feature.spectral_centroid(y=new_y, sr=new_sr))
                spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=new_y, sr=new_sr))
                spec_bw_var = np.var(librosa.feature.spectral_bandwidth(y=new_y, sr=new_sr))
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=new_y, sr=new_sr))
                rolloff_var = np.var(librosa.feature.spectral_rolloff(y=new_y, sr=new_sr))
                zcr = np.mean(librosa.feature.zero_crossing_rate(new_y))
                zcr_var = np.var(librosa.feature.zero_crossing_rate(new_y))
                harmony_y, puccsive_y = librosa.effects.hpss(new_y)
                harmony_mean = np.mean(librosa.effects.harmonic(harmony_y))
                harmony_var = np.var(librosa.effects.harmonic(harmony_y))
                perceptr_mean = np.mean(puccsive_y)
                perceptr_var = np.var(puccsive_y)
            
                
                datas.append({
                "path":data,
                "split":f"{j}_{k}",
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
                "zcr_var":zcr_var,
                "harmony_mean": harmony_mean,
                "harmony_var": harmony_var,
                "perceptr_mean": perceptr_mean,
                "perceptr_var" : perceptr_var
                })
                start += n
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