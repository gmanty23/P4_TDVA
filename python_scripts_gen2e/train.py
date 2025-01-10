import os
import pickle

import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
import model

from initialize_model import model_init

import matplotlib.pyplot as plt 

torch.set_default_dtype(torch.float32)

torch.manual_seed(0)

modelo = "model1" # Model to be trained

if modelo == "model1":
    latent_dims = 20
elif modelo == "model2":
    latent_dims = 4
else:
    assert "Modelo no encontrado"

vae = model.VariationalAutoencoder(latent_dims=latent_dims) # Llamamos al constructor y creamos un objeto VAE
#vae = model_init(vae,modelo)
#torch.save(vae.state_dict(), f"models/"+modelo+"/model.pth")
graph = torch.load(f"python_scripts_gen2e/models/{modelo}/model.pth",map_location=torch.device('cpu'))
#vae.load_state_dict(graph) # Cargamos los parámetros de la red (del modelo seleccionado anteriormente)

lr = 1e-4 # Learning rate
beta = 0.00001 # (hiper)Parámetro de ajuste de la VLB mediante la divergencia KL
num_epochs = 300 # Iteraciones de entrenamiento
batch_size = 8

prefix_name = 'multilateration_model_3DVae_4ls'

optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5) # Weight decay es un factor que multiplica a la norma L2 de los pesos de la red y se añade como término al cálculo del coste. 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

vae.to(device) # Movemos el modelo al dispositivo cuda (GPU) si disponemos de él; de lo contrario, trabajaremos en la CPU

def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train() # Sets the module in training mode => training mode?
    train_loss = 0.0
    reconstruction_error_epoch = 0.0
    kl_error_epoch = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x in tqdm(dataloader): # dataloader es un loader, y no un iterador. No obstante, es iterable. ¿Es implementar este for equivalente a hacer dataiter = iter(dataloader)?
        # Move tensor to the proper device
        x = x.to(device)
        # Check if incoming data has inf values => STOP
        if np.isinf(x.cpu().flatten().flatten()).any():
            hey = 1
        x_hat = vae(x) # El vae me está metiendo cuatro ventanas temporales más de las que tiene la señal de entrada (??)
        # Check if incoming data has nan values => STOP
        #x_hat = x_hat.cpu().numpy()
        # if np.isnan(x_hat.flatten().flatten()).any():
        #     hey = 1
        # Evaluate loss: Aquí hacemos el sumatorio de todos los resultados obtenidos para la ejecución de este batch
        reconstruction_error = ((torch.abs(x - x_hat)) ** 2).sum()
        kl_error = vae.encoder.kl.sum() * beta # Aquí vuelve a aparecer beta!
        loss = reconstruction_error + kl_error # Este coste es la VLB (que no hay que minimizar, sino maximizar, verdad?)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %s' % (loss.item().__str__()))
        # Acumulamos los resultados de todos los batches
        reconstruction_error_epoch += reconstruction_error.item()
        kl_error_epoch += kl_error.item()
        train_loss += loss.item()

    # Dividimos entre el tamaño del dataset para obtener los valores medios del coste de entrenamiento
    return train_loss / len(dataloader.dataset), reconstruction_error_epoch / len(dataloader.dataset), kl_error_epoch / len(dataloader.dataset)


### Testing function
def test_epoch(vae, device, dataloader, global_validation_loss):
    # Set evaluation mode for encoder and decoder
    vae.eval() # Sets VAE in evaluation (inference) mode. Suele ir acompañado del torch.no_grad (para no computar los gradientes)
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Decode data
            x_hat = vae(x)
            # loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            loss = ((torch.abs(x - x_hat)) ** 2).sum() + vae.encoder.kl.sum() * beta
            val_loss += loss.item()

    if global_validation_loss > val_loss / len(dataloader.dataset): # Nos estamos quedando con el mínimo del coste (¿caso peor por ser la VLB?)
        global_validation_loss = val_loss / len(dataloader.dataset)
        # torch.save(vae.state_dict(), f"checkpoints/{prefix_name}/model_checkpoint{global_validation_loss}.pth")



    return val_loss / len(dataloader.dataset), global_validation_loss


def load_spectrograms(spectrograms_path):
    x_train = []
    x_test = [] # Este array va a contener los nombres de los espectrogramas (¿Para qué?)

    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name).replace('\\','/')
            # ¿Por qué estas dos opciones de cargar los espectrogramas?
            if ".pkl" in file_path:
                with open(file_path, 'rb') as c: # 'rb' => read + binary
                    spectrogram = pickle.load(c)
            else:
                spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
            x_train.append(spectrogram[np.newaxis, ...]) # Esta sintaxis de spectrogram[np.newaxis, ...]???
            x_test.append(file_name)

    print(len(x_train))
    x_train = np.array(x_train) # Array numpy => (nº audios, dimensión vacía, modulo/fase, freq bins, time windows, L/R)

    return x_train, x_test 


x_train_L, _ = load_spectrograms("python_scripts_gen2e/inputs/spectrograms_train_L")
x_train_R, _ = load_spectrograms("python_scripts_gen2e/inputs/spectrograms_train_R")
x_train = np.concatenate((x_train_L,x_train_R),axis=2) # Array numpy => (nº audios, (modL phaL modR phaR), freq bins, time windows)
x_train = np.squeeze(x_train,axis=1)
#test_dataset, _ = load_spectrograms("python_scripts_gen2e/inputs/spectrograms_train") # spectrograms_test

train_loader = DataLoader(x_train.astype('float32'), batch_size=batch_size, shuffle=True)
#valid_loader = DataLoader(test_dataset.astype('float32'), batch_size=8, shuffle=True)
global_validation_loss = 1e9

train_losses = []
reconstruction_errors = []
kl_errors = []

for epoch in range(num_epochs):
   train_loss, reconstruction_error, kl_error = train_epoch(vae, device, train_loader, optimizer)
   #val_loss, global_validation_loss = test_epoch(vae, device, valid_loader, global_validation_loss)
   train_losses.append(train_loss)
   reconstruction_errors.append(reconstruction_error)
   kl_errors.append(kl_error)

   # Traza de época de entrenamiento
   print('\n EPOCH {}/{} \t train loss {:.3f} (recon error {:.3f}, kl error {:.3f}) \t'.format(epoch + 1,
                                                                                                              num_epochs,
                                                                                                              train_loss,
                                                                                                              reconstruction_error,
                                                                                                              kl_error))
   with open('progress.txt', 'a') as f: # Cambiar 'w' por 'a'
    f.write('\n EPOCH {}/{} \t train loss {:.3f} (recon error {:.3f}, kl error {:.3f}) \t'.format(epoch + 1,
                                                                                                              num_epochs,
                                                                                                              train_loss,
                                                                                                              reconstruction_error,
                                                                                                              kl_error))
    # Traza de época de validación
"""    print('\n EPOCH {}/{} \t train loss {:.3f} (recon error {:.3f}, kl error {:.3f}) \t val loss {:.3f}'.format(epoch + 1,
                                                                                                              num_epochs,
                                                                                                              train_loss,
                                                                                                              reconstruction_error,
                                                                                                              kl_error,
                                                                                                              val_loss)) """
    

torch.save(vae.state_dict(), f"python_scripts_gen2e/models/recurrente/"+modelo+"/model.pth")


plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", color='blue')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid()
plt.savefig("train_loss_curve.png") 
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(reconstruction_errors, label="Reconstruction Error", color='orange')
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Reconstruction Error Curve")
plt.legend()
plt.grid()
plt.savefig("reconstruction_error_curve.png")  

plt.figure(figsize=(8, 5))
plt.plot(kl_errors, label="KL Error", color='green')
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("KL Error Curve")
plt.legend()
plt.grid()
plt.savefig("kl_error_curve.png")  
plt.show()