import os
import pickle

#import librosa (FUNCIONES SUSTITUTIVAS)
    # Decibelios a amplitud => torchaudio.functional_db_to_amplitude(x, ref, power)
    # Normalizar señal de audio => signal./numpy.linalg.norm(signal,axis=0)
# from scipy.signal.windows import kaiser
# import scipy.signal
# scipy.signal.kaiser = kaiser
import mdct
import numpy as np
# import scipy
# scipy.real = np.real

import soundfile as sf
import stft
import torch
from sklearn import preprocessing
from tqdm import tqdm

import model
#from initialize_model import model_init # IMPORTANTE: Importamos función model_init!!


# Sirve para cargar los espectrogramas junto con su path
def load_spectrograms(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if ".pkl" in file_path:
                with open(file_path, 'rb') as c: # 'rb' => read + binary
                    spectrogram = pickle.load(c)
            else:
                spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    return x_train, file_paths


# ESTA HAY QUE MODIFICARLA PARA ENSAMBLAR AUDIO ESTÉREO A PARTIR DE CADA UNA / CREAR generate_stereo_from_spectrogram que use generate_audio_from_spectrogram
def generate_audio_from_spectrogram(predicted_data, min_max_L, min_max_R, name):
    array_module_L = (predicted_data[0, 0, ...] - 0) / (1 - 0) # ¿Por qué se está empleando la elipsis (...) aquí? ¿Y lo de restar 0?
    array_module_L = array_module_L * (min_max_L['max_mod'] - min_max_L['min_mod']) + min_max_L['min_mod'] # Esto es una especie de desnormalización, ¿verdad?
    array_phase_L = (predicted_data[0, 1, ...] - 0) / (1 - 0)
    array_phase_L = array_phase_L * (min_max_L['max_pha'] - min_max_L['min_pha']) + min_max_L['min_pha']
    array_module_R = (predicted_data[1, 0, ...] - 0) / (1 - 0)
    array_module_R = array_module_R * (min_max_R['max_mod'] - min_max_R['min_mod']) + min_max_R['min_mod']
    array_phase_R = (predicted_data[1, 1, ...] - 0) / (1 - 0)
    array_phase_R = array_phase_R * (min_max_R['max_pha'] - min_max_R['min_pha']) + min_max_R['min_pha']
    # Build stereo signal!
    array_module = np.concatenate((array_module_L[np.newaxis,...],array_module_R[np.newaxis,...]),axis=0)
    array_phase = np.concatenate((array_phase_L[np.newaxis,...],array_phase_R[np.newaxis,...]),axis=0)

    spc_phase = (array_phase + 2 * np.pi) % (2 * np.pi) # ¿Por qué estamos tomando la fase simétrica con respecto a pi?
    spc_phase = np.cos(spc_phase) + 1j * np.sin(spc_phase) # Construimos el complejo de módulo uno y la fase adecuada
    # reconstructed = librosa.db_to_amplitude(array_module) * spc_phase # Escalamos el complejo anterior por el módulo adecuado para
    reconstructed = 10**(array_module/20) * spc_phase

    # stft_settings
    reconstructed = np.swapaxes(np.swapaxes(reconstructed,0,2),0,1)
    signal = mdct.fast.imclt(reconstructed, # (?) Asumo que se conocen los parámetros con los que se ha calculado la MCLT directa para obtener los espectrogramas de entrada al modelo, ¿cierto?
                             framelength=1024,
                             hopsize=512,
                             overlap=2, # MCLT
                             centered=True,
                             window=stft.stft.cosine,
                             padding=0,
                             #outlength=32722)
                             #outlength=32722*4)
                             outlength=176000)
    # Re-normalización
    sigMax, sigMin = signal.max(),signal.min()
    signal = (signal-sigMin)/(sigMax-sigMin)


    # Guardado de archivos generados 
    name = os.path.split(name)[-1] + ".wav"
    #save_path = os.path.join("outputs", modelo, predecir, name)
    save_path = os.path.join("python_scripts_gen2e","outputs",modelo,"tonyHawkProSkaterIIThursday", name)

    sf.write(save_path, signal, 48000)


# %% Decidir modelo
modelo = "model1"

if modelo == "model1":
    latent_dims = 20
elif modelo == "model2":
    latent_dims = 4
else:
    assert "Modelo no encontrado"


#%% Carga de espectrogramas

# Aquí me quedo con los datos de train y test,
# luego trabajo normalmente con test pero se pueden probar ambos
'''train_dataset_L, train_paths_L = load_spectrograms("python_scripts_gen2e\inputs\spectrograms_fireball_L_measured")
train_paths_L = [os.path.split(i)[-1].split('.pkl')[0] for i in train_paths_L] # ¿Le estamos quitando la extensión de archivo a las rutas de los espectrogramas?
test_dataset_L, test_paths_L = load_spectrograms("python_scripts_gen2e\inputs\spectrograms_fireball_L_measured")
test_paths_L = [os.path.split(i)[-1].split('.pkl')[0] for i in test_paths_L]
train_dataset_R, train_paths_R = load_spectrograms("python_scripts_gen2e\inputs\spectrograms_fireball_R_measured")
train_paths_R = [os.path.split(i)[-1].split('.pkl')[0] for i in train_paths_R]
test_dataset_R, test_paths_R = load_spectrograms("python_scripts_gen2e\inputs\spectrograms_fireball_R_measured")
test_paths_R = [os.path.split(i)[-1].split('.pkl')[0] for i in test_paths_R]
'''
test_dataset_L, test_paths_L = load_spectrograms(r"python_scripts_gen2e\inputs\tonyHawkProSkaterIIThursdayL")
test_paths_L = [os.path.split(i)[-1].split('.pkl')[0] for i in test_paths_L]
test_dataset_R, test_paths_R = load_spectrograms(r"python_scripts_gen2e\inputs\tonyHawkProSkaterIIThursdayR")
test_paths_R = [os.path.split(i)[-1].split('.pkl')[0] for i in test_paths_R]
#dvfzvf
# %% Carga del modelo de autoencoder variacional

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Se inicializan los modelos. Si cambias al de 4 dimensiones, hay que cambiar tanto path como latent_dims
vae = model.VariationalAutoencoder(latent_dims=latent_dims)
#vae = model_init(vae,modelo)
graph = torch.load(f"python_scripts_gen2e/models/fireballEntrenoBaseSinLogica4/"+modelo+"/model.pth",map_location=torch.device('cpu'))
vae.load_state_dict(graph)
vae.to(device)
vae.eval() # Modo evaluación (no se pueden cambiar parámetros)


# %% Carga de valores mínimo y máximo originales de la señal para su posterior desnormalización

# Aquí recupero los valores min max de fase y amplitud
with open(r"python_scripts_gen2e/inputs/min_max_test_fireball/min_max_values.pkl", "rb") as input_file: # ¿Qué aspecto tienen las entradas de min_max_values.pkl?
    min_max_dict = pickle.load(input_file) # Es min_max_dict un diccionario que contiene los valores máximo y mínimo de cada espectrograma??
    min_max_dict = {os.path.split(k)[-1]: v for k, v in min_max_dict.items()}

# %%

predecir = "test"

if predecir == "train":
    loop = zip(train_dataset_L,train_paths_L,train_dataset_R,train_paths_R) # zip() devuelve un iterador de tuplas (de dos elementos, en este caso, espectrograma + path correspondiente)
elif predecir == "test":
    loop = zip(test_dataset_L,test_paths_L,test_dataset_R,test_paths_R)
else:
    assert "Error en el subconjunto a predecir"

# Descomentar si estamos usando solo datos de elevación 0
#predecir = "train_E0"

# %%

latent_space = []

with torch.no_grad():  # No need to track the gradients
    for xL, pathL, xR, pathR in tqdm(loop): # La generación de registros la hace espectrograma a espectrograma (no agrupados en tensores)
        nameL = os.path.split(pathL)[-1]
        nameR = os.path.split(pathR)[-1]
        nameR = 'fireball_E-10_A180_SNR49_R'
        nameL = 'fireball_E-10_A180_SNR49_L'
        min_max_L = min_max_dict[nameL]
        min_max_R = min_max_dict[nameR]
        xL = xL[np.newaxis, ...]
        xR = xR[np.newaxis, ...]
        # Ensambla xR y xL para crear x (entrada al vae) -> Primero canales L, luego canales R
        x = np.concatenate((xL,xR),axis=1)
        # Move tensor to the proper device
        x = torch.from_numpy(x).to(device)
        # Encode data
        latent_space.append(vae.encoder(x.double()).cpu().detach().numpy())
        predicted_data = vae(x.double()).cpu().detach().numpy()
        pred_L = predicted_data[:,0:2,:,:]
        pred_R = predicted_data[:,2:4,:,:]
        predicted_data = np.concatenate((pred_L,pred_R),axis=0)
        name = '_'.join(nameL.split('_')[0:-1])
        generate_audio_from_spectrogram(predicted_data, min_max_L, min_max_R, name)

with open(f'python_scripts_gen2e/models/fireballEntrenoBaseSinLogica4/"+modelo+"/latent_spaces.npy', 'wb') as f:
    np.save(f, latent_space)
