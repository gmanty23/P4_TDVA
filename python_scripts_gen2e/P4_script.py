# import os
# import pickle
# import model
# import numpy as np
# import torch

# def load_spectrograms(spectrograms_path):
#     x_train = []
#     x_test = [] # Este array va a contener los nombres de los espectrogramas (¿Para qué?)

#     for root, _, file_names in os.walk(spectrograms_path):
#         for file_name in file_names:
#             file_path = os.path.join(root, file_name).replace('\\','/')
#             # ¿Por qué estas dos opciones de cargar los espectrogramas?
#             if ".pkl" in file_path:
#                 with open(file_path, 'rb') as c: # 'rb' => read + binary
#                     spectrogram = pickle.load(c)
#             else:
#                 spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
#             x_train.append(spectrogram[np.newaxis, ...]) # Esta sintaxis de spectrogram[np.newaxis, ...]???
#             x_test.append(file_name)

#     print(len(x_train))
#     x_train = np.array(x_train) # Array numpy => (nº audios, dimensión vacía, modulo/fase, freq bins, time windows, L/R)

#     return x_train, x_test 

# #Inicialización del vae
# modelo = "model2" # Model to be trained
# if modelo == "model1":
#     latent_dims = 20
# elif modelo == "model2":
#     latent_dims = 4
# else:
#     assert "Modelo no encontrado"
# vae = model.VariationalAutoencoder(latent_dims=latent_dims) # Llamamos al constructor y creamos un objeto VAE

# #Carga de los espectogramas
# x_train_L, _ = load_spectrograms("python_scripts_gen2e/inputs/spectrograms_train_L")
# x_train_R, _ = load_spectrograms("python_scripts_gen2e/inputs/spectrograms_train_R")
# x = np.concatenate((x_train_L,x_train_R),axis=2) # Array numpy => (nº audios, (modL phaL modR phaR), freq bins, time windows)
# x = np.squeeze(x,axis=1)


# graph = torch.load(f"python_scripts_gen2e/models/{modelo}/model.pth",map_location=torch.device('cpu'))
# vae.load_state_dict(graph) # Cargamos los parámetros de la red (del modelo seleccionado anteriormente)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f'Selected device: {device}')
# vae.to(device) # Movemos el modelo al dispositivo cuda (GPU) si disponemos de él; de lo contrario, trabajaremos en la CPU

# vae.eval() # Sets VAE in evaluation (inference) mode. Suele ir acompañado del torch.no_grad (para no computar los gradientes)
# with torch.no_grad(): # No need to track the gradients
#     # Move tensor to the proper device
#     x = x.to(device)
#     # Decode data
#     x_hat = vae(x)

import torch

def check_cuda_and_pytorch_version():
    print("Versión de PyTorch instalada:", torch.__version__)
    print("Versión de CUDA utilizada por PyTorch:", torch.version.cuda)
    print("PyTorch detecta GPU:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("Nombre de la GPU:", torch.cuda.get_device_name(0))
        print("Número de GPUs disponibles:", torch.cuda.device_count())
    else:
        print("No se detectó ninguna GPU. Verifica la instalación de CUDA y los drivers de NVIDIA.")

if __name__ == "__main__":
    check_cuda_and_pytorch_version()


