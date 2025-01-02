import os
import pickle

#import librosa
import mdct
import numpy as np
import soundfile as sf
import stft
import torch
from sklearn import preprocessing
from tqdm import tqdm
import scipy

import model

# Sirve para cargar los espectrogramas junto con su path

def extract_mclt(signal, frame_length, hop_length):
    signal = signal + 1e-8
    mclt_trans = mdct.fast.mclt(signal, framelength=frame_length, hopsize=hop_length)[:, :-1] # Extraer MCLT, dependencia mdct: https://mdct.readthedocs.io/en/latest/modules/mdct.html
    #spectrogram, phase = librosa.magphase(mclt_trans) # Computar espectrograma MCLT y su fase (la MCLT es compleja)
    spectrogram = np.abs(mclt_trans)
    angles = np.angle(mclt_trans) # Computar el ángulo a partir de la fase
    angles2pi = (angles + 2 * np.pi) % (2 * np.pi) # Normalizar el ángulo entre 0 y 2pi
    unwrapped_phase = np.unwrap(angles2pi) # Desenrollar la fase (este es un truquito para que no haya saltos, y la fase sea continua)
    #log_spectrogram = librosa.amplitude_to_db(spectrogram) # Calcular decibelios a partir del espectrograma: librosa
    log_spectrogram = 20*np.log10(spectrogram)
    
    return log_spectrogram, unwrapped_phase, mclt_trans # Devolvemos todo esto, la idea es luego normalizar por separado magnitud y fase.

def load_spectrograms(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if ".pkl" in file_path:
                with open(file_path, 'rb') as c:
                    spectrogram = pickle.load(c)
            else:
                spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
                x_train.append(spectrogram)
                file_paths.append(file_path)
    x_train = np.array(x_train)
    #x_train = x_train[..., np.newaxis]  # -> (3000, 256, 64, 1)
    return x_train, file_paths

def generate_audio_from_spectrogram(predicted_data, min_max, name, save_path):
    array_module = (predicted_data[0, 0, ...] - 0) / (1 - 0) # Desnormalización entre 0 y 1
    array_module = array_module * (min_max['max_mod'] - min_max['min_mod']) + min_max['min_mod'] # Usamos las constantes de normalización de la magnitud
    array_phase = (predicted_data[0, 1, ...] - 0) / (1 - 0) # Desnormalización entre 0 y 1
    array_phase = array_phase * (min_max['max_pha'] - min_max['min_pha']) + min_max['min_pha'] # Usamos las constantes de normalización de la fase

    spc_phase = (array_phase + 2 * np.pi) % (2 * np.pi) # Desenrrollamos los ángulos (realmente esto no hace falta hacerlo)
    spc_phase = np.cos(spc_phase) + 1j * np.sin(spc_phase) # Pasamos de ángulo a fase
    #reconstructed = librosa.db_to_amplitude(array_module) * spc_phase
    reconstructed = 10**(array_module/20) * spc_phase # Reconstruimos el espectrograma MCLT

    # stft_settings
    signal = mdct.fast.imclt(reconstructed, # Calculamos la transformada inversa
                             framelength=1024,  # Hay que pasarle el frame_length original
                             hopsize=512, # Hop_length original
                             overlap=2, # Esto debería ser la división entre frame_length y hop_length
                             centered=True, # Esto debería coincidir con la mclt original pero por defecto es True
                             window=stft.stft.cosine, # Esto debería coincidir con la mclt original pero suele ser stft.stft.cosine: dependencia https://stft.readthedocs.io/en/latest/stft.html
                             padding=0, # Esto debería coincidir con la mclt original pero por defecto es True
                             outlength=32722) # Este parámetro es super tricky... tiene que ver con la longitud de la señal de entrada. Para los audios que hemos usado es este valor. Para otros audios, hay que mirar en el objeto MCLT que crea la librería al computar la mclt. Está medio oculto... si vais a probar con otros audios lo miramos juntos.
    #signal = librosa.util.normalize(signal) # Normalizamos la señal para que los valores de amplitud vayan entre 0 y 1
    signal = np.expand_dims(signal,0)
    signal = preprocessing.normalize(signal,norm="max") # norm=max equivale a normalizar entre -1 y 1 (la definición de norma es norma infinito)
    signal = np.squeeze(signal)
    
    name = os.path.split(name)[-1] + ".wav"
    save_path = os.path.join(save_path, name)
    sf.write(save_path, signal, 44100) # guardamos

def load_wavs(wavs_path):
    dataset = []
    file_paths = []
    Fs = 0
    for root, _, file_names in os.walk(wavs_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name).replace("\\","/")
            data = scipy.io.wavfile.read(file_path, mmap=False)  # (n_bins, n_frames, 1)
            dataset.append(data[1])
            Fs = data[0] # Quedará la frecuencia de muestreo del último audio extraido (todos los audios deben tener la misma Fs)
            file_paths.append(file_path)
    dataset = np.array(dataset)
    #x_train = x_train[..., np.newaxis]  # -> (3000, 256, 64, 1)
    return dataset, file_paths, Fs

def serialize(save_path, name, x_modpha):
    pickle_out = open(os.path.join(save_path, name, ".pkl"), "wb")
    pickle.dump(x_modpha, pickle_out) 
    pickle_out.close()

# %% MAIN

caseRoute = [["measured","E0"],["interpolated","E0"],["measured","A0"],["interpolated","A0"]]
# Inicializamos el diccionario de valores máximos y mínimos (que más tarde guardaremos)
min_max_dict = dict()
min_max_path_file = r"inputs/min_max_test/min_max_values.pkl" # Path donde guardaremos los valores min-max de los espectrogramas generados

for i in caseRoute:
    case = i[0]
    route = i[1]

    # Esto se configura solo (no tocar)
    wavs_path = r"waterdrop_billinear/"+route+"_only/"+case
    spectrograms_path = r"inputs/spectrograms_"+route

    #%% Carga de espectrogramas

    # Aquí me quedo con los datos de train y test,
    # luego trabajo normalmente con test pero se pueden probar ambos
    dataset, paths, Fs = load_wavs(wavs_path)
    paths = [os.path.split(i)[-1].split('.wav')[0] for i in paths]

    mclt_framelength = 1024
    mclt_hopsize = 512

    # %% Carga del modelo de autoencoder variacional

    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    # %%
    loop = zip(dataset, paths)

    for x, path in tqdm(loop): 
            name = os.path.split(path)[-1]
            # Move tensor to the proper device
            x = torch.from_numpy(x).to(device)
            # Calculamos espectrograma (MCLT)
            x_mod, x_pha, x_trans = extract_mclt(x, mclt_framelength, mclt_hopsize)
            max_mod = np.amax(x_mod,axis=(0,1))
            min_mod = np.amin(x_mod,axis=(0,1))
            max_pha = np.amax(x_pha,axis=(0,1))
            min_pha = np.amin(x_pha,axis=(0,1))
            x_min_max = {
                "max_mod": max_mod,
                "min_mod": min_mod,
                "max_pha": max_pha,
                "min_pha": min_pha
            }
            x_min_max_L = {
                "max_mod": max_mod[0],
                "min_mod": min_mod[0],
                "max_pha": max_pha[0],
                "min_pha": min_pha[0]
            }
            x_min_max_R = {
                "max_mod": max_mod[1],
                "min_mod": min_mod[1],
                "max_pha": max_pha[1],
                "min_pha": min_pha[1]
            }
            # min_max_dict tiene TODOS los nombres de los espectrogramas y sus valores extremos
            min_max_dict[name+"_L"] = x_min_max_L
            min_max_dict[name+"_R"] = x_min_max_R
            # Normalizar módulo y fase
            x_mod = (x_mod - x_min_max["min_mod"]) / (x_min_max["max_mod"] - x_min_max["min_mod"])
            x_pha = (x_pha - x_min_max["min_pha"]) / (x_min_max["max_pha"] - x_min_max["min_pha"])
            # Creamos nuevo eje y concatenamos (creamos tensor con un canal módulo y otro fase)
            x_mod = x_mod[np.newaxis, :] 
            x_pha = x_pha[np.newaxis, :]
            x_modpha_L = np.concatenate((x_mod[:,:,:,0], x_pha[:,:,:,0]), axis=0) # (modulo/fase, freq bins, time windows)
            x_modpha_R = np.concatenate((x_mod[:,:,:,1], x_pha[:,:,:,1]), axis=0)
            #x_modpha = np.concatenate((x_mod, x_pha), axis=2)
            # Guardamos los espectrogramas
            name_L = os.path.split(name)[-1] + "_L.pkl"
            name_R = os.path.split(name)[-1] + "_R.pkl"
            # Guardamos espectrograma L
            spectrograms_path_stereo = spectrograms_path+"_L_"+case
            save_path = os.path.join(spectrograms_path_stereo, name_L).replace('\\','/')
            with open(save_path, 'wb') as f:
                pickle.dump(x_modpha_L, f)
            # Guardamos espectrograma R
            spectrograms_path_stereo = spectrograms_path+"_R_"+case
            save_path = os.path.join(spectrograms_path_stereo, name_R).replace('\\','/')
            with open(save_path, 'wb') as f:
                pickle.dump(x_modpha_R, f)
            #save_path = os.path.join(spectrograms_path, name).replace('\\','/')
            #with open(save_path, 'wb') as f:
            #    pickle.dump(x_modpha, f)

# Guardamos los valores máximos y mínimos de los espectrogramas
with open(min_max_path_file, 'wb') as output_file:
    pickle.dump(min_max_dict, output_file) 
        


