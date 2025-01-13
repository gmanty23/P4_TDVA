# import os
# import pickle

# import numpy as np
# import torch

# from tqdm import tqdm
# from torch.utils.data import DataLoader
# import model

# def model_init(vae,modelo):

#     torch.set_default_dtype(torch.double)
#     torch.manual_seed(0)

#     # Dimensiones latentes en función del modelo (pasado como parámetro)
#     if modelo == "model1":
#         latent_dims = 20
#     elif modelo == "model2":
#         latent_dims = 4
#     else:
#         assert "Modelo no encontrado"

#     modeloL = modelo + "-L"
#     modeloR = modelo + "-R"
#     modelo3 = modelo + "-3"

#     # Carga de pesos e inicialización de instancia vae
#     graphL = torch.load(f"python_scripts_gen2e/models/{modeloL}/model.pth",map_location=torch.device('cpu'))
#     graphR = torch.load(f"python_scripts_gen2e/models/{modeloR}/model.pth",map_location=torch.device('cpu'))
#     graph3 = torch.load(f"python_scripts_gen2e/models/{modelo3}/model.pth",map_location=torch.device('cpu'))

#     # Cargamos en los pesos del vae con los de modelx-L y modelx-R: https://discuss.pytorch.org/t/loading-a-specific-layer-from-checkpoint/52725
#     with torch.no_grad():
#         # Encoder conv1 -> La mitad de los pesos son ceros para mantener los canales separados a lo largo (profundo!) de la red
#         # La primera mitad de los canales debería tener sólo info del canal L: la segunda, del canal R
#         # La segunda dimensión son canales de entrada (primera mitad L, segunda mitad R), la primera son canales de salida (la mitad serán ceros)
#         vae.encoder.conv1.weight[0:512,0:2,:,:].copy_(graphL.get('encoder.conv1.weight'))
#         vae.encoder.conv1.weight[0:512,2:4,:,:] = torch.zeros(vae.encoder.conv1.weight[0:512,2:4,:,:].size())
#         vae.encoder.conv1.weight[512:1024,0:2,:,:] = torch.zeros(vae.encoder.conv1.weight[512:1024,0:2,:,:].size())
#         vae.encoder.conv1.weight[512:1024,2:4,:,:].copy_(graphR.get('encoder.conv1.weight'))
#         vae.encoder.conv1.bias[0:512].copy_(graphL.get('encoder.conv1.bias')) # Asignamos desde el 0 hasta el 511 (sintaxis rara!)
#         vae.encoder.conv1.bias[512:1024].copy_(graphR.get('encoder.conv1.bias')) # Asignamos desde el 512 hasta el 1024 (sintaxis rara!)
#         # Encoder batch1
#         vae.encoder.batch1.weight[0:512].copy_(graphL.get('encoder.batch1.weight'))
#         vae.encoder.batch1.weight[512:1024].copy_(graphR.get('encoder.batch1.weight'))
#         vae.encoder.batch1.bias[0:512].copy_(graphL.get('encoder.batch1.bias'))
#         vae.encoder.batch1.bias[512:1024].copy_(graphR.get('encoder.batch1.bias'))
#         vae.encoder.batch1.running_mean[0:512].copy_(graphL.get('encoder.batch1.running_mean'))
#         vae.encoder.batch1.running_mean[512:1024].copy_(graphR.get('encoder.batch1.running_mean'))
#         vae.encoder.batch1.running_var[0:512].copy_(graphL.get('encoder.batch1.running_var'))
#         vae.encoder.batch1.running_var[512:1024].copy_(graphR.get('encoder.batch1.running_var'))
#         # Encoder conv2
#         vae.encoder.conv2.weight[0:256,0:512,:,:].copy_(graphL.get('encoder.conv2.weight'))
#         vae.encoder.conv2.weight[0:256,512:1024,:,:] = torch.zeros(vae.encoder.conv2.weight[0:256,512:1024,:,:].size())
#         vae.encoder.conv2.weight[256:512,0:512,:,:] = torch.zeros(vae.encoder.conv2.weight[256:512,0:512,:,:].size())
#         vae.encoder.conv2.weight[256:512,512:1024,:,:].copy_(graphR.get('encoder.conv2.weight'))
#         vae.encoder.conv2.bias[0:256].copy_(graphL.get('encoder.conv2.bias'))
#         vae.encoder.conv2.bias[256:512].copy_(graphR.get('encoder.conv2.bias'))
#         # Encoder batch 2
#         vae.encoder.batch2.weight[0:256].copy_(graphL.get('encoder.batch2.weight'))
#         vae.encoder.batch2.weight[256:512].copy_(graphR.get('encoder.batch2.weight'))
#         vae.encoder.batch2.bias[0:256].copy_(graphL.get('encoder.batch2.bias'))
#         vae.encoder.batch2.bias[256:512].copy_(graphR.get('encoder.batch2.bias'))
#         vae.encoder.batch2.running_mean[0:256].copy_(graphL.get('encoder.batch2.running_mean'))
#         vae.encoder.batch2.running_mean[256:512].copy_(graphR.get('encoder.batch2.running_mean'))
#         vae.encoder.batch2.running_var[0:256].copy_(graphL.get('encoder.batch2.running_var'))
#         vae.encoder.batch2.running_var[256:512].copy_(graphR.get('encoder.batch2.running_var'))
#         # Encoder conv3
#         vae.encoder.conv3.weight[0:128,0:256,:,:].copy_(graphL.get('encoder.conv3.weight'))
#         vae.encoder.conv3.weight[0:128,256:512,:,:] = torch.zeros(vae.encoder.conv3.weight[0:128,256:512,:,:].size())
#         vae.encoder.conv3.weight[128:256,0:256,:,:] = torch.zeros(vae.encoder.conv3.weight[128:256,0:256,:,:].size())
#         vae.encoder.conv3.weight[128:256,256:512,:,:].copy_(graphR.get('encoder.conv3.weight'))
#         vae.encoder.conv3.bias[0:128].copy_(graphL.get('encoder.conv3.bias'))
#         vae.encoder.conv3.bias[128:256].copy_(graphR.get('encoder.conv3.bias'))
#         # Encoder batch3
#         vae.encoder.batch3.weight[0:128].copy_(graphL.get('encoder.batch3.weight'))
#         vae.encoder.batch3.weight[128:256].copy_(graphR.get('encoder.batch3.weight'))
#         vae.encoder.batch3.bias[0:128].copy_(graphL.get('encoder.batch3.bias'))
#         vae.encoder.batch3.bias[128:256].copy_(graphR.get('encoder.batch3.bias'))
#         vae.encoder.batch3.running_mean[0:128].copy_(graphL.get('encoder.batch3.running_mean'))
#         vae.encoder.batch3.running_mean[128:256].copy_(graphR.get('encoder.batch3.running_mean'))
#         vae.encoder.batch3.running_var[0:128].copy_(graphL.get('encoder.batch3.running_var'))
#         vae.encoder.batch3.running_var[128:256].copy_(graphR.get('encoder.batch3.running_var'))
#         # Encoder conv4
#         vae.encoder.conv4.weight[0:64,0:128,:,:].copy_(graphL.get('encoder.conv4.weight'))
#         vae.encoder.conv4.weight[64:128,0:128,:,:] = torch.zeros(vae.encoder.conv4.weight[64:128,0:128,:,:].size())
#         vae.encoder.conv4.weight[0:64,128:256,:,:] = torch.zeros(vae.encoder.conv4.weight[0:64,128:256,:,:].size())
#         vae.encoder.conv4.weight[64:128,128:256,:,:].copy_(graphR.get('encoder.conv4.weight'))
#         vae.encoder.conv4.bias[0:64].copy_(graphL.get('encoder.conv4.bias'))
#         vae.encoder.conv4.bias[64:128].copy_(graphR.get('encoder.conv4.bias'))
#         # Encoder batch4
#         vae.encoder.batch4.weight[0:64].copy_(graphL.get('encoder.batch4.weight'))
#         vae.encoder.batch4.weight[64:128].copy_(graphR.get('encoder.batch4.weight'))
#         vae.encoder.batch4.bias[0:64].copy_(graphL.get('encoder.batch4.bias'))
#         vae.encoder.batch4.bias[64:128].copy_(graphR.get('encoder.batch4.bias'))
#         vae.encoder.batch4.running_mean[0:64].copy_(graphL.get('encoder.batch4.running_mean'))
#         vae.encoder.batch4.running_mean[64:128].copy_(graphR.get('encoder.batch4.running_mean'))
#         vae.encoder.batch4.running_var[0:64].copy_(graphL.get('encoder.batch4.running_var'))
#         vae.encoder.batch4.running_var[64:128].copy_(graphR.get('encoder.batch4.running_var'))
#         # Encoder conv5
#         vae.encoder.conv5.weight[0:32,0:64,:,:].copy_(graphL.get('encoder.conv5.weight'))
#         vae.encoder.conv5.weight[32:64,0:64,:,:] = torch.zeros(vae.encoder.conv5.weight[32:64,0:64,:,:].size())
#         vae.encoder.conv5.weight[0:32,64:128,:,:] = torch.zeros(vae.encoder.conv5.weight[0:32,64:128,:,:].size())
#         vae.encoder.conv5.weight[32:64,64:128,:,:].copy_(graphR.get('encoder.conv5.weight'))
#         vae.encoder.conv5.bias[0:32].copy_(graphL.get('encoder.conv5.bias'))
#         vae.encoder.conv5.bias[32:64].copy_(graphR.get('encoder.conv5.bias'))
#         # Encoder batch5
#         vae.encoder.batch5.weight.copy_(graph3.get('encoder.batch4.weight'))
#         vae.encoder.batch5.bias.copy_(graph3.get('encoder.batch4.bias'))
#         vae.encoder.batch5.running_mean.copy_(graph3.get('encoder.batch4.running_mean'))
#         vae.encoder.batch5.running_var.copy_(graph3.get('encoder.batch4.running_var'))
#         # Encoder conv6
#         vae.encoder.conv6.weight.copy_(graph3.get('encoder.conv5.weight'))
#         vae.encoder.conv6.bias.copy_(graph3.get('encoder.conv5.bias'))
#         # Encoder mu
#         vae.encoder.mu.weight.copy_(graph3.get('encoder.mu.weight'))
#         vae.encoder.mu.bias.copy_(graph3.get('encoder.mu.bias'))
#         # Encoder var
#         vae.encoder.var.weight.copy_(graph3.get('encoder.var.weight'))
#         vae.encoder.var.bias.copy_(graph3.get('encoder.var.bias'))

#         # Decoder linear layer 1
#         vae.decoder.decoder_lin._modules['0'].weight.copy_(graph3.get('decoder.decoder_lin.0.weight'))
#         vae.decoder.decoder_lin._modules['0'].bias.copy_(graph3.get('decoder.decoder_lin.0.bias'))
#         # Decoder linear layer 2
#         vae.decoder.decoder_lin._modules['2'].weight.copy_(graph3.get('decoder.decoder_lin.2.weight'))
#         vae.decoder.decoder_lin._modules['2'].bias.copy_(graph3.get('decoder.decoder_lin.2.bias'))
#         # Decoder dec0
#         vae.decoder.dec0.weight.copy_(graph3.get('decoder.dec1.weight'))
#         vae.decoder.dec0.bias.copy_(graph3.get('decoder.dec1.bias'))
#         # Decoder batch0
#         vae.decoder.batch0.weight.copy_(graph3.get('decoder.batch1.weight'))
#         vae.decoder.batch0.bias.copy_(graph3.get('decoder.batch1.bias'))
#         vae.decoder.batch0.running_mean.copy_(graph3.get('decoder.batch1.running_mean'))
#         vae.decoder.batch0.running_var.copy_(graph3.get('decoder.batch1.running_var'))
#         # Decoder dec1
#         vae.decoder.dec1.weight[0:32,0:64,:,:].copy_(graphL.get('decoder.dec1.weight'))
#         vae.decoder.dec1.weight[0:32,64:128,:,:] = torch.zeros(vae.decoder.dec1.weight[0:32,64:128].size())
#         vae.decoder.dec1.weight[32:64,0:64,:,:] = torch.zeros(vae.decoder.dec1.weight[32:64,0:64].size())
#         vae.decoder.dec1.weight[32:64,64:128,:,:].copy_(graphR.get('decoder.dec1.weight'))
#         vae.decoder.dec1.bias[0:64].copy:(graphL.get('decoder.dec1.bias'))
#         vae.decoder.dec1.bias[64:128].copy:(graphR.get('decoder.dec1.bias'))
#         # Decoder batch1
#         vae.decoder.batch1.weight[0:64].copy_(graphL.get('decoder.batch1.weight'))
#         vae.decoder.batch1.weight[64:128].copy_(graphR.get('decoder.batch1.weight'))
#         vae.decoder.batch1.bias[0:64].copy_(graphL.get('decoder.batch1.bias'))
#         vae.decoder.batch1.bias[64:128].copy_(graphR.get('decoder.batch1.bias'))
#         vae.decoder.batch1.running_mean[0:64].copy_(graphL.get('decoder.batch1.running_mean'))
#         vae.decoder.batch1.running_mean[64:128].copy_(graphR.get('decoder.batch1.running_mean'))
#         vae.decoder.batch1.running_var[0:64].copy_(graphL.get('decoder.batch1.running_var'))
#         vae.decoder.batch1.running_var[64:128].copy_(graphR.get('decoder.batch1.running_var'))
#         # Decoder dec2
#         vae.decoder.dec2.weight[0:64,0:128,:,:].copy_(graphL.get('decoder.dec2.weight'))
#         vae.decoder.dec2.weight[0:64,128:256,:,:] = torch.zeros(vae.decoder.dec2.weight[0:64,128:256].size())
#         vae.decoder.dec2.weight[64:128,0:128,:,:] = torch.zeros(vae.decoder.dec2.weight[64:128,0:128].size())
#         vae.decoder.dec2.weight[64:128,128:256,:,:].copy_(graphR.get('decoder.dec2.weight'))
#         vae.decoder.dec2.bias[0:128].copy:(graphL.get('decoder.dec2.bias'))
#         vae.decoder.dec2.bias[128:256].copy:(graphR.get('decoder.dec2.bias'))
#         # Decoder batch2
#         vae.decoder.batch2.weight[0:128].copy_(graphL.get('decoder.batch2.weight'))
#         vae.decoder.batch2.weight[128:256].copy_(graphR.get('decoder.batch2.weight'))
#         vae.decoder.batch2.bias[0:128].copy_(graphL.get('decoder.batch2.bias'))
#         vae.decoder.batch2.bias[128:256].copy_(graphR.get('decoder.batch2.bias'))
#         vae.decoder.batch2.running_mean[0:128].copy_(graphL.get('decoder.batch2.running_mean'))
#         vae.decoder.batch2.running_mean[128:256].copy_(graphR.get('decoder.batch2.running_mean'))
#         vae.decoder.batch2.running_var[0:128].copy_(graphL.get('decoder.batch2.running_var'))
#         vae.decoder.batch2.running_var[128:256].copy_(graphR.get('decoder.batch2.running_var'))
#         # Decoder dec3
#         vae.decoder.dec3.weight[0:128,0:256,:,:].copy_(graphL.get('decoder.dec3.weight'))
#         vae.decoder.dec3.weight[0:128,256:512,:,:] = torch.zeros(vae.decoder.dec3.weight[0:128,256:512,:,:].size())
#         vae.decoder.dec3.weight[128:256,0:256,:,:] = torch.zeros(vae.decoder.dec3.weight[128:256,0:256,:,:].size())
#         vae.decoder.dec3.weight[128:256,256:512,:,:].copy_(graphR.get('decoder.dec3.weight'))
#         vae.decoder.dec3.bias[0:256].copy:(graphL.get('decoder.dec3.bias'))
#         vae.decoder.dec3.bias[256:512].copy:(graphR.get('decoder.dec3.bias'))
#         # Decoder batch3
#         vae.decoder.batch3.weight[0:256].copy_(graphL.get('decoder.batch3.weight'))
#         vae.decoder.batch3.weight[256:512].copy_(graphR.get('decoder.batch3.weight'))
#         vae.decoder.batch3.bias[0:256].copy_(graphL.get('decoder.batch3.bias'))
#         vae.decoder.batch3.bias[256:512].copy_(graphR.get('decoder.batch3.bias'))
#         vae.decoder.batch3.running_mean[0:256].copy_(graphL.get('decoder.batch3.running_mean'))
#         vae.decoder.batch3.running_mean[256:512].copy_(graphR.get('decoder.batch3.running_mean'))
#         vae.decoder.batch3.running_var[0:256].copy_(graphL.get('decoder.batch3.running_var'))
#         vae.decoder.batch3.running_var[256:512].copy_(graphR.get('decoder.batch3.running_var'))
#         # Decoder dec4
#         vae.decoder.dec4.weight[0:256,0:512,:,:].copy_(graphL.get('decoder.dec4.weight'))
#         vae.decoder.dec4.weight[0:256,512:1024,:,:] = torch.zeros(vae.decoder.dec4.weight[0:256,512:1024,:,:].size())
#         vae.decoder.dec4.weight[256:512,0:512,:,:] = torch.zeros(vae.decoder.dec4.weight[256:512,0:512,:,:].size())
#         vae.decoder.dec4.weight[256:512,512:1024,:,:].copy_(graphR.get('decoder.dec4.weight'))
#         vae.decoder.dec4.bias[0:512].copy:(graphL.get('decoder.dec4.bias'))
#         vae.decoder.dec4.bias[512:1024].copy:(graphR.get('decoder.dec4.bias'))
#          # Decoder batch4
#         vae.decoder.batch4.weight[0:512].copy_(graphL.get('decoder.batch4.weight'))
#         vae.decoder.batch4.weight[512:1024].copy_(graphR.get('decoder.batch4.weight'))
#         vae.decoder.batch4.bias[0:512].copy_(graphL.get('decoder.batch4.bias'))
#         vae.decoder.batch4.bias[512:1024].copy_(graphR.get('decoder.batch4.bias'))
#         vae.decoder.batch4.running_mean[0:512].copy_(graphL.get('decoder.batch4.running_mean'))
#         vae.decoder.batch4.running_mean[512:1024].copy_(graphR.get('decoder.batch4.running_mean'))
#         vae.decoder.batch4.running_var[0:512].copy_(graphL.get('decoder.batch4.running_var'))
#         vae.decoder.batch4.running_var[512:1024].copy_(graphR.get('decoder.batch4.running_var'))
#         # Decoder dec5
#         vae.decoder.dec5.weight[0:512,0:2,:,:].copy_(graphL.get('decoder.dec5.weight'))
#         vae.decoder.dec5.weight[0:512,2:4,:,:] = torch.zeros(vae.decoder.dec5.weight[0:512,2:4,:,:].size())
#         vae.decoder.dec5.weight[512:1024,0:2,:,:] = torch.zeros(vae.decoder.dec5.weight[512:1024,0:2,:,:].size())
#         vae.decoder.dec5.weight[512:1024,2:4,:,:].copy_(graphR.get('decoder.dec5.weight'))
#         vae.decoder.dec5.bias[0:2].copy:(graphL.get('decoder.dec5.bias'))
#         vae.decoder.dec5.bias[2:4].copy:(graphR.get('decoder.dec5.bias'))

#     return vae

# #%% MAIN
# # Selección de modelo
# modelo = "model1"
# if modelo == "model1":
#     latent_dims = 20
# elif modelo == "model2":
#     latent_dims = 4
# else:
#     assert "Modelo no encontrado"
# # Instanciamos VAE
# vae = model.VariationalAutoencoder(latent_dims=latent_dims) # Llamamos al constructor y creamos un objeto VAE
# # Inicializamos pesos
# model_init(vae,modelo)
# # Guardamos los parámetros del modelo (extendido con respecto a los VAEs de la gen1)
# torch.save(vae.state_dict(), f"python_scripts_gen2e/models/"+modelo+"/model.pth")
# torch.save(vae.state_dict(), f"python_scripts_gen2e/models/"+modelo+"/model.pth")
# torch.save(vae.state_dict(), f"python_scripts_gen2e/models/"+modelo+"/model.pth")
