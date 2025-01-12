import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.cuda.is_available)

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VariationalEncoder(nn.Module):

    def __init__(self, latent_dim):
        super(VariationalEncoder, self).__init__() # Llamamos al constructor del padre (clase nn.Module) // Repetir pregunta: parámetros de super()

        # # encoder
        # self.conv1 = nn.Conv2d(in_channels=2*2, out_channels=512*2, kernel_size=(5, 5), stride=(2, 2)) # Voy a concatenar los datos de entrada en la dimension de bines de frecuencia (512->1024)... a ver qué tal!
        # self.batch1 = nn.BatchNorm2d(512*2) # BatchNorm2d => Preguntar por la expresión de la documentación de PyTorch
        # self.conv2 = nn.Conv2d(in_channels=512*2, out_channels=256*2, kernel_size=(3, 3), stride=(2, 2))
        # self.batch2 = nn.BatchNorm2d(256*2)
        # self.conv3 = nn.Conv2d(in_channels=256*2, out_channels=128*2, kernel_size=(3, 3), stride=(2, 2))
        # self.batch3 = nn.BatchNorm2d(128*2)
        # self.conv4 = nn.Conv2d(in_channels=128*2, out_channels=64*2, kernel_size=(2, 2), stride=(2, 2))
        # self.batch4 = nn.BatchNorm2d(64*2)
        # self.conv5 = nn.Conv2d(in_channels=64*2, out_channels=32*2, kernel_size=(1, 1), stride=(1, 1))
        # self.batch5 = nn.BatchNorm2d(32*2)
        # self.conv6 = nn.Conv2d(in_channels=32*2, out_channels=32, kernel_size=(1, 1), stride=(1, 1))

        # LSTM para capturar dependencias temporales
        # La entrada será de tamaño (batch_size, secuencia_temporal, 4 * 512) = (batch_size, 57, 2048)
        self.lstm = nn.LSTM(input_size = 4*512, hidden_size = 256,num_layers= 1, batch_first=True)

        # distribution parameters
        # self.mu = nn.Linear(32 * 31 * 3, latent_dim)
        # self.var = nn.Linear(32 * 31 * 3, latent_dim)

        self.mu = nn.Linear(256, latent_dim)
        self.var = nn.Linear(256, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        #self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        #self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        # x = F.relu(self.batch1(self.conv1(x))) # Capa 1
        # x = F.relu(self.batch2(self.conv2(x))) # Capa 2
        # x = F.relu(self.batch3(self.conv3(x))) # Capa 3
        # x = F.relu(self.batch4(self.conv4(x))) # Capa 4
        # x = F.relu(self.batch5(self.conv5(x))) # Capa 5
        # x = self.conv6(x)                      # Capa 6 
        # x = torch.flatten(x, start_dim=1) # Convertimos a un vector unidimensional
        #mu = self.mu(x)

        # Aplanar las dimensiones de canales y frecuencias (4, 512) => 2048
        x = x.view(x.size(0), x.size(3), -1)  # Forma (batch_size, secuencia_temporal, 2048)
        # Procesar la secuencia temporal con LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
         # Usar la salida del último paso de tiempo para obtener mu y sigma
        h_n = h_n[-1] 
        mu = self.mu(h_n)                        # Capa '7'
        sigma = torch.exp(self.var(h_n)).to(device)

        z = self.N.sample(mu.shape).to(device)      # Capa '7'
        z = mu + sigma*z#self.N.sample(mu.shape) # ¿Qué es mu.shape? => Tamaño multidimensional de la media (mu)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum() # Y esta forma de calcular la divergencia KL? => Es la expresión analítica de la dkl para dos gaussianas 
        
        return z
    
# Esto para cargar el valor de x del modelo de la gen1
#import numpy as np
#x1 = x.numpy()
#with open(f'temp.npy', 'rb') as f:
#    x2 = np.load(f)
#print(np.all(x1[:,0:int(x1.shape[1]/2),:,:]==x2))
#print(np.sum(x1[:,0:int(x1.shape[1]/2),:,:]-x2))


class Decoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()

        # # Capa lineal para reconstruir la salida de la LSTM
        # self.decoder_lin = nn.Sequential( # Asumo que nn.Sequential() define una serie de funciones que se aplicarán una tras otra 
        #     nn.Linear(latent_dims, 128),
        #     nn.ReLU(True), # ¿Qué significa el parámetro True en esta función ReLU?
        #     nn.Linear(128, 2*2*512*57),
        #     nn.ReLU(True)
        # )

         # self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 31, 3))

        # self.dec0 = nn.ConvTranspose2d(in_channels=32, out_channels=32*2, kernel_size=(1, 1), stride=(1, 1))
        # self.batch0 = nn.BatchNorm2d(32*2)
        # self.dec1 = nn.ConvTranspose2d(in_channels=32*2, out_channels=64*2, kernel_size=(1, 1), stride=(1, 1)) # ¿ConvTranspose2d?
        # self.batch1 = nn.BatchNorm2d(64*2)
        # self.dec2 = nn.ConvTranspose2d(in_channels=64*2, out_channels=128*2, kernel_size=(2, 2), stride=(2, 2))
        # self.batch2 = nn.BatchNorm2d(128*2)
        # self.dec3 = nn.ConvTranspose2d(in_channels=128*2, out_channels=256*2, kernel_size=(3, 3), stride=(2, 2), output_padding=1)
        # self.batch3 = nn.BatchNorm2d(256*2)
        # self.dec4 = nn.ConvTranspose2d(in_channels=256*2, out_channels=512*2, kernel_size=(3, 3), stride=(2, 2), output_padding=1)
        # self.batch4 = nn.BatchNorm2d(512*2)
        # self.dec5 = nn.ConvTranspose2d(in_channels=512*2, out_channels=2*2, kernel_size=(5, 5), stride=(2, 2), output_padding=1)

        # Capa LSTM para procesar la representación latente
        self.lstm = nn.LSTM(latent_dims, hidden_size=256, num_layers= 1,batch_first=True)

        # Capa lineal para mapear a un tensor inicial interpretable
        #self.lin = nn.Linear(256, 256 * 32 * 8)  # Salida inicial: (512 canales, 32x8 tamaño)
        self.lin = nn.Linear(256, 4 * 512 * 176)  # Salida inicial: (512 canales, 32x8 tamaño)
        # Decodificación con convoluciones transpuestas
        '''        self.resize = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (32x8) -> (64x16)
            nn.ReLU(),
            #Capa 2: (64x16) -> (128x32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64x16) -> (128x32)
            nn.ReLU(),
            #Capa 3: (128x32) -> (256x64)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # (128x32) -> (256x64)
            nn.ReLU(),
            #Capa 4: (256x64) -> (512x128)
            nn.ConvTranspose2d(32, 4, kernel_size=(4,1), stride=(2,3), padding=(1,7)),   # (128x32) -> (512x176)
            nn.ReLU(),
            #Capa 5: (512x128) -> (512x174)
            #nn.ConvTranspose2d(32, 4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            #nn.Upsample(size=(512, 174), mode='bilinear', align_corners=False),
            #nn.ReLU(),
            
            #nn.ConvTranspose2d(64, 4, kernel_size=(4,3), stride=(2,1), padding=1)  # (256x64) -> (512x64)
        )'''


       

    def forward(self, x):
        # x = self.decoder_lin(x)
        # x = self.unflatten(x)
        # x = F.relu(self.batch0(self.dec0(x)))
        # x = F.relu(self.batch1(self.dec1(x)))
        # x = F.relu(self.batch2(self.dec2(x)))
        # x = F.relu(self.batch3(self.dec3(x)))
        # x = F.relu(self.batch4(self.dec4(x)))
        # x = self.dec5(x)
        # x = torch.sigmoid(x) # ¿Y esta última función de activación? ¿Es acaso para normalizar el resultado?
        # return x

        # x tiene la forma (batch_size, latent_dim), necesitamos convertirlo en una secuencia de longitud 57.

        # Añadir una dimensión de secuencia
        x = x.unsqueeze(1) #(batch_size, 1, latent_dim)
        
        # Pasar por el LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len=1, hidden_size=256)
        lstm_out = lstm_out[:, -1, :]  # Seleccionamos la última salida de la secuencia (batch_size, hidden_size)   

        # Transformar con capa lineal
        x = self.lin(lstm_out)  # (batch_size, 512 * 16 * 4)
        x = x.view(-1, 4, 512, 176)  # Reorganizar a (batch_size, canales, altura, ancho)

        # Refinar con convoluciones transpuestas
        #x = self.resize(x)  # Salida final: (batch_size, 4, 512, 64)

        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device) # Señal de entrada
        z = self.encoder(x) # Al encoder
        return self.decoder(z) # al decoder