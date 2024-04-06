import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import config


class ClickbaitDataset(Dataset):

    def __init__(self, ds, tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer = tokenizer

        #insere os tokens especiais, bos,eos, pad e cls em que cls é o token de classe
        #self.cls_token = torch.tensor([tokenizer.token_to_id("[CLS]")], dtype=torch.int64)
        #acredito que consigo usar o BOS token para fazer a mesma coisa que o cls token faria
        self.bos_token = torch.tensor([tokenizer.token_to_id("[BOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data_line = self.ds[idx]
        src_text = data_line['text']
        label = data_line["label"]


        # Transform the text into tokens
        enc_input_tokens = self.tokenizer.encode(src_text).ids


        # Add sos, eos and padding to each sentence
        num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <bos> and <eos>


        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if num_padding_tokens < 0 :
             raise ValueError("Sentence is too long")
        # Add <bos> and <eos> token - BOS will serve as a CLS token as well
        encoder_input = torch.cat(
            [
                self.bos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )


        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len


        return {
            "encoder_input": encoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "label": label,  # (seq_len)
            "text": src_text,
        }


