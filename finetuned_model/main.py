import torch
import numpy as np
import matplotlib . pyplot as plt
import csv
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate

#------Preparação do dataset
non_clickbait_dire = 'data\\non_clickbait_data.csv'
clickbait_dire = 'data\clickbait_data.csv'

def read_and_label_csv(name, label):
    data = []
    with open(name, 'r', encoding='utf-8') as arquivo_csv:
        # Cria um leitor CSV
        leitor_csv = csv.reader(arquivo_csv)

        # Itera sobre as linhas do arquivo CSV e as armazena na lista
        for linha in leitor_csv:
            if len(linha)==0:
                continue
            else:
                 data.append(','.join(linha))
    return pd.DataFrame({"text":data,"label":label})


# n_clickbait = read_and_label_csv(non_clickbait_dire,0)
# clickbait = read_and_label_csv(clickbait_dire,1)
# final_data = pd.concat([clickbait, n_clickbait], ignore_index=True)
# final_data = final_data.sample(frac=1, random_state=19)
# final_data.to_csv('data\\final_data.csv', index=False)
# final_data = pd.read_csv('data\\final_data.csv')
#
# split = int(len(final_data) * 0.8)  # 80% para treino
#
# train_data = final_data.iloc[:split]
# train_data.to_csv('data\\train_data.csv', index=False)
# validation_data = final_data.iloc[split:]
# validation_data.to_csv('data\\validation_data.csv', index=False)

train_data = Dataset.from_csv('data\\train_data.csv')
val_data = Dataset.from_csv('data\\validation_data.csv')

test = {"train": train_data , "val": val_data}
final_data = DatasetDict(test)


def plot_occurencies(data,titulo):
    contagem_valores = data['label'].value_counts()
    # Criação do histograma
    plt.bar(contagem_valores.index.astype(str), contagem_valores)

    # Adiciona o número total de ocorrências em cada barra
    for indice, valor in enumerate(contagem_valores):
        plt.text(indice, valor, str(valor), ha='center', va='bottom')
    plt.xlabel('Labels')
    plt.ylabel('Valores')
    plt.title(titulo)

model_checkpoint = 'distilbert-base-uncased'
#model_checkpoint = 'roberta-base' # you can alternatively use roberta-base but this model is bigger thus training will take longer

# define label maps
id2label = {0: "Non  clickbait", 1: "Clickbait"}
label2id = {"Non  clickbait":0, "Clickbait":1}

# generate classification model from model_checkpoint
model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)


# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    # extract text
    text = examples["text"]

    #tokenize and truncate text

    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
       text,
        return_tensors="np",
        truncation=True,
        max_length=128
    )

    return tokenized_inputs
tokenized_dataset = final_data.map(tokenize_function, batched=True)

#Save the tokenized dataset
# for nome, dataset in tokenized_dataset.items():
#     # Converter o conjunto de dados para DataFrame do pandas
#     df = dataset.to_pandas()
#
#     # Salvar o DataFrame como arquivo CSV
#     nome_arquivo = f'{nome}.csv'  # Nome do arquivo baseado no nome do dataset
#     df.to_csv(nome_arquivo, index=False)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")
# define an evaluation function to pass into trainer later

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}


text_list = ["Which 2016 World Tour Should You See Based On Your Birth Month",
             "Canada pursues new nuclear research reactor to produce medical isotopes",
             "6 Untrue Facts About Mental Health You Probably Believe",
             "21 Call Center Horror Stories That'll Give You Nightmares",
             "Cuban talk show accuses U.S. diplomat of helping anti-government groups"]


peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=10,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ["query","key","value"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_peft_model(model, peft_config).to(device)
print(model)
model.print_trainable_parameters()

# hyperparameters
lr = 1e-3
batch_size = 4
num_epochs = 10

# define training arguments
training_args = TrainingArguments(
    output_dir= model_checkpoint + "-lora-clikbait",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# creater trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    tokenizer=tokenizer,
    data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)


#trainer.train()


# model.to('cpu') # moving to mps for Mac (can alternatively do 'cpu')
#
# print("Trained model predictions:")
# print("--------------------------")
# for text in text_list:
#     inputs = tokenizer.encode(text, return_tensors="pt").to("cpu") # moving to mps for Mac (can alternatively do 'cpu')
#
#     logits = model(inputs).logits
#     predictions = torch.max(logits,1).indices
#
#     print(text + " - " + id2label[predictions.tolist()[0]])