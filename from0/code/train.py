import torch
import pandas as pd

import transformer
from data import ClickbaitDataset
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import config
from config import get_weights_file_path
import transformer as model

import torchmetrics
import wandb
import os
from torch.utils.tensorboard import SummaryWriter

class readDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # Used in the begining to lowercase the dataset
        # self.data['word_count'] = self.data["text"].apply(lambda x: len(x.split()))
        # self.data = self.data[self.data['word_count'] <= 150]
        # self.data.drop(columns=['word_count'], inplace=True)
        #self.data['text'] = self.data['text'].apply(lambda x: x.lower())
        # self.data.to_csv(csv_file,index=False)

    def __len__(self):
        return len(self.data)
    def numtokes(self):
        max_len = 0
        for x in self.data["text"]:
            max_len = max(max_len, len(x.split()))
        return max_len
    def plot_ocorencia(self):
        lengths = [len(text.split()) for text in self.data["text"]]
        plt.hist(lengths, bins=range(min(lengths), max(lengths) + 1), edgecolor='black')
        plt.xlabel('Número de palavras',fontsize=14)
        plt.ylabel('Ocorrência',fontsize=14)
        plt.title('Ocorrência de Comprimentos de Frase',fontsize=16)
        plt.grid(True)
        plt.show()
    def plot_labels(self):
        counts = self.data['label'].value_counts()

        # Criando o gráfico de barras
        plt.bar(counts.index, counts.values)

        # Adicionando rótulos
        plt.title('Distribuição por Classe',fontsize=16)
        plt.xlabel('Classe',fontsize=14)
        plt.ylabel('Frequência',fontsize=14)

        for i, value in enumerate(counts.values):
            plt.text(i, value + 0.1, str(value), ha='center',fontsize=14)

        # Definindo os rótulos do eixo x
        plt.xticks(counts.index, ['Não Sensacionalista', 'Clickbait'],fontsize=14)
        plt.show()

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        return sample
def get_all_sentences(ds):
    for item in ds:
        yield item['text']

def get_or_build_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def count_words(text):
    words = text.split()
    return len(words)
def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = readDataset(config["dataset_path"])

    #finds the max seq_len in the dataset
    max_len = ds_raw.numtokes()
    print(f'Max length of source sentence: {max_len}')

    #plots the ocurrencie
    #ds_raw.plot_ocorencia()

    # Build tokenizers
    tokenizer = get_or_build_tokenizer(config, ds_raw)

    #
    # # Keep 80% for training, 20% for validation
    train_ds_size = int(0.8 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = ClickbaitDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = ClickbaitDataset(val_ds_raw, tokenizer, config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    # rows in the dataset -> encoder_input,encoder_mask,label,text
    return train_dataloader, val_dataloader, tokenizer

def get_model(config, vocab_size):
    model = transformer.build_transformer(vocab_size, config['seq_len'], d_model=config['d_model'],d_ff=config["dff"],N=config["num_of_blocks"],
    dropout=config["dropout"], num_class=config["num_class"], h=config["num_heads"])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['dataset']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_weights = '{:,.0f}'.format(num_weights).replace(',', '.')
    print(model)
    print(f"Total of {num_weights} trainable parameters")
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    #eps avoids crash by divising by 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = config.latest_weights_file_path(config) if preload == 'latest' else config.get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')


    loss_fn =nn.CrossEntropyLoss().to(device)
    #training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)


            # Run the tensors through the encoder, decoder and the projection layer
            proj_output = model.forward(encoder_input, encoder_mask) # (B, seq_len, num_class)

            # Compare the output with the label
            label = batch['label'].to(device) # (Bach,1)


            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, config["num_class"]), label.view(-1))
            #print("LOSS:",loss.item())
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss for plot
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)   #zero the grads

            global_step += 1


        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, device, lambda msg: batch_iterator.write(msg), global_step)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        #saves the optimizer for resume epochs
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


def run_validation(model, validation_ds, device, print_msg, global_step, num_examples=2):
    model.eval()
    count = 0


    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            input = batch["encoder_input"].to(device)  # (b, seq_len)
            mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = torch.argmax( torch.softmax(model.forward(input, mask),dim=-1) ,dim=-1) #(Output clas)

            source_text = batch["text"][0]
            #retrive the class output
            classes = ["Non-Clickbait", "Clickbait"]



            # Print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'PREDICTED: ':>12}{classes[model_out]}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

def accuracy(model,val_data,device):
    model.eval()
    right_prep = 0
    with torch.no_grad():
        for batch in val_data:
            input = batch["encoder_input"].to(device)  # (b, seq_len)
            mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)
            label = batch["label"].to('cpu')        #(batch)

            model_out = torch.argmax(torch.softmax(model.forward(input, mask), dim=-1), dim=-1)  # (Output clas)
            if model_out == label:
                right_prep+=1
        return right_prep/len(val_data)

def query(model,phrase: str,tokenizer,seq_len,device):
    model.eval()
    input = tokenizer.encode(phrase.lower()).ids
    bos_token = torch.tensor([tokenizer.token_to_id("[BOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    num_padding_tokens = seq_len - len(input) - 2

    encoder_input = torch.cat([bos_token,
                               torch.tensor(input, dtype=torch.int64),eos_token,
                               torch.tensor([pad_token] * num_padding_tokens,
                                            dtype=torch.int64)],dim=0).unsqueeze(0)

    mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()
    output = model.forward(encoder_input,mask).to(device)
    predict = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
    classes = ["Non-Clickbait","Clickbait"]
    return classes[predict]

if __name__ == '__main__':
    config = config.get_config()
    config['preload'] = None
    test = readDataset(config["dataset_path"])
    #to see the bos token
    #-----------------------Inferring:---------------------------
    # train_dataloader, val_dataloader, tokenizer = get_ds(config)
    # model = get_model(config, tokenizer.get_vocab_size()).to('cpu')
    # print(tokenizer.get_vocab_size())
    test.plot_ocorencia()

    # saves = torch.load('click_bait_weights\\weights-config1\\tmodel_09.pt',map_location=torch.device('cpu'))
    # model.load_state_dict(saves['model_state_dict'])
    # model.eval()
    #
    #
    #
    # sentence = "I was going to present my project and this happened"
    # predict = query(model,sentence,tokenizer,config["seq_len"],"cpu")
    # print(sentence, "->",predict)
    #---------------------------------------------
    # num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('{:,.0f}'.format(num_weights).replace(',', '.'))

    # examples = next(iter(train_dataloader))
    # print(examples['label'].view(-1))
    # print(tokenizer.get_vocab(),tokenizer.get_vocab_size())
    # ------------------Validation:---------------------------
    #run_validation(model, val_dataloader, 'cpu', lambda msg: print(msg), 0, num_examples=5)