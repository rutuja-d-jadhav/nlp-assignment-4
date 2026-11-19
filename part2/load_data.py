import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
LT_TOKEN = "LESSTHAN"

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''

        self.tokenizer = T5TokenizerFast.from_pretrained(
            "google-t5/t5-small"
        )

        self.split = split

        self.data = self.process_data(
            data_folder,
            split,
            self.tokenizer
        )

    def process_data(self, data_folder, split, tokenizer):

        # Read natural language inputs
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_lines = load_lines(nl_path)

        # Test set has no ground-truth SQL
        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_lines = load_lines(sql_path)

            assert len(nl_lines) == len(sql_lines)

        processed_data = []

        for i, nl in enumerate(nl_lines):

            # 1. Tokenize encoder input
            encoder_ids = tokenizer.encode(
                nl,
                add_special_tokens=True
            )

            # T5 uses the PAD token as its decoder start token
            initial_decoder_input = [tokenizer.pad_token_id]

            if split == "test":

                processed_data.append((
                    torch.tensor(encoder_ids, dtype=torch.long),
                    torch.tensor(initial_decoder_input, dtype=torch.long)
                ))

            else:

                sql = sql_lines[i]
                sql = sql.replace("<", LT_TOKEN) # NEW: '<' → placeholder before tokenizing

                # 2. Tokenize SQL target
                decoder_targets = tokenizer.encode(
                    sql,
                    add_special_tokens=True
                )

                # 3. Shift right to create decoder inputs
                decoder_inputs = (
                    [tokenizer.pad_token_id]
                    + decoder_targets[:-1]
                )

                processed_data.append((
                    torch.tensor(encoder_ids, dtype=torch.long),
                    torch.tensor(decoder_inputs, dtype=torch.long),
                    torch.tensor(decoder_targets, dtype=torch.long),
                    torch.tensor(initial_decoder_input, dtype=torch.long)
                ))

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''



    encoder_inputs = [item[0] for item in batch]
    decoder_inputs = [item[1] for item in batch]
    decoder_targets = [item[2] for item in batch]
    initial_decoder_inputs = [item[3] for item in batch]

    # Pad encoder inputs
    encoder_ids = pad_sequence(
        encoder_inputs,
        batch_first=True,
        padding_value=PAD_IDX
    )

    # Create encoder attention mask
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # Pad decoder inputs
    decoder_inputs = pad_sequence(
        decoder_inputs,
        batch_first=True,
        padding_value=PAD_IDX
    )

    # Pad decoder targets
    decoder_targets = pad_sequence(
        decoder_targets,
        batch_first=True,
        padding_value=PAD_IDX
    )

    # All examples start decoding with the PAD token
    initial_decoder_inputs = torch.stack(
        initial_decoder_inputs
    )

    return (
        encoder_ids,
        encoder_mask,
        decoder_inputs,
        decoder_targets,
        initial_decoder_inputs
    )

    #return [], [], [], [], []

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''



    encoder_inputs = [item[0] for item in batch]
    initial_decoder_inputs = [item[1] for item in batch]

    encoder_ids = pad_sequence(
        encoder_inputs,
        batch_first=True,
        padding_value=PAD_IDX
    )

    encoder_mask = (encoder_ids != PAD_IDX).long()

    initial_decoder_inputs = torch.stack(
        initial_decoder_inputs
    )

    return (
        encoder_ids,
        encoder_mask,
        initial_decoder_inputs
    )


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x