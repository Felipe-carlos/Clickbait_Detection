import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math

loss_train_config1 =pd.read_csv("click_bait_weights\weights-config1\\tmodel.csv")
loss_train_config2 =pd.read_csv("click_bait_weights\weights-config2\\config2.csv")

loss_config1 =pd.read_csv("metricas_config1.csv")
loss_config2 =pd.read_csv("metricas_config2.csv")


def smooth_line(window_size,data):
        return  data.rolling(window_size, min_periods=1).mean()
epochs = np.linspace(0, 10, 10)
epochs_10 = np.linspace(0, 10, 10)

plt.figure()
# for num in [1,2]:
#     nome_dados = globals()[f"loss_config{num}"]
#     plt.plot(epochs,nome_dados['Loss'],label=f'Pre-trained, config {num}',linestyle='-')
#     plt.xlabel('Épocas')
#     plt.ylabel('Perdas')
#     plt.title('Va')
#     plt.grid(True)
#     # linha = smooth_line(25,nome_dados)
#     # plt.plot(epochs, linha, color='red', label='Tedency line')




loss_train_finetuned_r_25 = [0.2195,0.1624,0.1293,0.1157,0.0977,0.0844,0.0803,0.0737,0.0637,0.0568]
loss_val_finetuned_r_25 = [0.158101,0.1900,0.1363,0.11752,0.118079,0.14164,0.1332,0.1456,0.1373,0.1396]
acuraccy_val_finetuned_r_25 = [0.945831370701837,0.94049301, 0.9555660229235359,0.9605903595540901,0.9626314963102528,0.9601193279949757,0.9646726330664155,0.9621604647511384,0.9643586120270058,0.9656146961846443]



loss_train_finetuned_r_5 = [0.2535,0.1521,0.1181,0.0926,0.0782,0.0678,0.0527,0.0446,0.0357,0.0298]
loss_val_finetuned_r_5 = [0.1793,0.1945,0.1058,0.102126,0.110532,0.153470,0.1389,0.1519,0.1453,0.1583]
acuraccy_val_finetuned_r_5 =  [0.9384518762757105,0.940650023551578,0.9649866541058251,0.9660857277437588,0.9690689276181504, 0.9629455173496624, 0.970325011775789, 0.9684408855393312, 0.9722091380122468, 0.9711100643743131]

print(loss_config1)
print(loss_config2)
#------------------------------------------------------
# plt.figure()
# plt.plot(epochs, loss_val_finetuned_r_25,label='Fine-tuned/r = 25',linestyle='-.',color='b')
# plt.plot(epochs,loss_val_finetuned_r_5,label='Fine-tuned/r = 5',linestyle='-.',color='r')
# plt.plot(epochs,loss_config1["Loss"],label='Pré-treinado/config 1',linestyle='-',color='k')
# plt.plot(epochs,loss_config2["Loss"],label='Pré-treinado/config 2',linestyle='-',color='m')
# plt.xlabel('Épocas',fontsize=14)
# plt.ylabel('Perdas',fontsize=14)
# plt.title('Perdas de Validação ',fontsize=16)
# plt.grid(True)
# plt.legend( fontsize=12)
#
#
# plt.figure()
# # Plot do segundo gráfico no segundo subplot
# plt.plot(epochs,[x*100 for x in acuraccy_val_finetuned_r_25] ,label='Fine-tuned/r = 25',linestyle='-.',color='b')
# plt.plot(epochs,[x*100 for x in acuraccy_val_finetuned_r_5],label='Fine-tuned/r = 5',linestyle='-.',color='r')
# plt.plot(epochs,[x*100 for x in loss_config1["Acc"]],label='Pré-treinado/config 1',linestyle='-',color='k')
# plt.plot(epochs,[x*100 for x in loss_config2["Acc"]],label='Pré-treinado/config 2',linestyle='-',color='m')
# plt.xlabel('Épocas',fontsize=14)
# plt.ylabel('Acurácia',fontsize=14)
# plt.title('Acurácia de Validação',fontsize=16)
# plt.grid(True)
# plt.legend( fontsize=12)
#


# Mostrar os gráficos
# plt.show()

#------------------------------------------------------
# plt.figure()
#
# plt.plot(epochs, loss_val_finetuned_r_25,label='Fine-tuned/r = 25',linestyle='-.',color='b')
# plt.plot(epochs,loss_val_finetuned_r_5,label='Fine-tuned/r = 5',linestyle='-.',color='r')
# plt.plot(epochs,loss_config1["Loss"],label='Pré-treinado/config 1',linestyle='-',color='k')
# plt.plot(epochs,loss_config2["Loss"],label='Pré-treinado/config 2',linestyle='-',color='m')
# plt.xlabel('Épocas',fontsize=14)
# plt.ylabel('Perdas',fontsize=14)
# plt.title('Perdas através do treinamento',fontsize=16)
# plt.grid(True)
# plt.legend( fontsize=12)
# plt.show()
# #
# plt.plot(epochs_10,loss_config1["Loss"],label='Validação/config 1',linestyle='--',color='k')
# plt.plot(epochs_10,loss_config2["Loss"],label='Validação/config 2',linestyle='--',color='m')
# plt.plot(epochs,smooth_line(200,loss_train_config1["Value"]),label='Treino/config 1',linestyle='-',color='k')
# plt.plot(epochs,smooth_line(200,loss_train_config2["Value"]),label='Treino/config 2',linestyle='-',color='m')
# plt.xlabel('Épocas',fontsize=14)
# plt.ylabel('Perdas',fontsize=14)
# plt.title('Perdas através do treinamento',fontsize=16)
# plt.grid(True)
# plt.legend( fontsize=12)
# plt.show()






# x = np.linspace(0, 10, 10)
# plt.figure()
# for num in [25, 5]:
#     nome_dados = globals()[f"loss_val_finetuned_r_{num}"]
#     plt.plot(x, nome_dados, label=f'r= {num}')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Validation loss')
#     plt.grid(True)
#     #linha = smooth_line(25, nome_dados)
#    # plt.plot(epochs, linha, color='red', label='Tedency line')
# plt.legend()
# plt.ylim(0.04, 0.26)
# plt.show()