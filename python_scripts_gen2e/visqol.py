import subprocess
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
ref_path = r"C:\Users\ferma\Documents\#DirectorioDeFer\Documentos académicos\Universidad\4º\GAPS\TFG\Code\Kemar_HRTF\python_scripts\inputs_wav\wavs_train_padd"
deg_path1 = r"C:\Users\ferma\Documents\#DirectorioDeFer\Documentos académicos\Universidad\4º\GAPS\TFG\Code\Kemar_HRTF\python_scripts\outputs\model1-S\train_padd"
deg_path2 = r"C:\Users\ferma\Documents\#DirectorioDeFer\Documentos académicos\Universidad\4º\GAPS\TFG\Code\Kemar_HRTF\python_scripts\outputs\model2-S\train_padd"


# Ubicación de ViSQOL (ejecutable y modelo)
visqol_executable = r"C:/Users/ferma/Documents/#DirectorioDeFer/Documentos académicos/Universidad/4º/GAPS/TFG/Code/Kemar_HRTF/visqol2/visqol/bazel-bin/visqol.exe"
model = r"C:/Users/ferma/Documents/#DirectorioDeFer/Documentos académicos/Universidad/4º/GAPS/TFG/Code/Kemar_HRTF/visqol2/visqol/model/libsvm_nu_svr_model.txt"

# Bucle para evaluar todos los audios
scores1 = []
scores2 = []
for _, _, file_names in os.walk(ref_path):

    for file_name in file_names:
        ref_file = os.path.join(ref_path,file_name).replace('\\','/')
        deg_file1 = os.path.join(deg_path1,file_name).replace('\\','/')
        deg_file2 = os.path.join(deg_path2,file_name).replace('\\','/')
        # Compute ViSQOL
        result1 = subprocess.run([visqol_executable, "--reference_file", ref_file, "--degraded_file", deg_file1, "--similarity_to_quality_model", model], capture_output=True, text=True)
        result2 = subprocess.run([visqol_executable, "--reference_file", ref_file, "--degraded_file", deg_file2, "--similarity_to_quality_model", model], capture_output=True, text=True)

        # Obtain score (LQO)
        visqol_score1 = float(result1.stdout.split('LQO:')[-1].strip())
        scores1 = np.append(scores1,visqol_score1)
        visqol_score2 = float(result2.stdout.split('LQO:')[-1].strip())
        scores2 = np.append(scores2,visqol_score2)

# Calcular media y desviación estándar
mu1 = np.mean(scores1)
sigma1 = np.std(scores1)
mu2 = np.mean(scores2)
sigma2 = np.std(scores2)
# sigma = np.sqrt(np.mean(np.abs(scores-mu)**2)) # Bajo nivel!

# Histograma
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
n_bins = 20
# Model 1
axs[0].hist(scores1, bins=n_bins)
axs[0].set_title("ViSQOL for Model 1"), axs[0].set_xlabel("LQO")
# Model 2
axs[1].hist(scores2, bins=n_bins)
axs[1].set_title("ViSQOL for Model 2"), axs[1].set_xlabel("LQO")

plt.show()
