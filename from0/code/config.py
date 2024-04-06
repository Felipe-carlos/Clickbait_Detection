from pathlib import Path

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 20,
        "lr": 10**-5,
        "seq_len": 200,
        "d_model": 256,
        "num_heads":8,
        "num_of_blocks":8,
        "dff":1024,
        "dropout":0.1,
        "dataset": "click_bait",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/tmodel",
        "num_class": 2,
        "dataset_path":'data\\final_data.csv'
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['dataset']}_{config['model_folder']}"
    print(model_folder)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['dataset']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

