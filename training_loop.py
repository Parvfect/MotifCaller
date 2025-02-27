
from nn import MotifCaller
from training_data import (
    load_training_data, data_preproc,
    create_label_for_training
)
from Levenshtein import ratio
from sklearn.preprocessing import normalize
import pandas as pd
from greedy_decoder import GreedyCTCDecoder
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import CTCLoss
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import get_actual_transcript, get_savepaths
import numpy as np
from typing import List, Dict
import random
from model_config import ModelConfig
from training_monitoring import wandb_login, start_wandb_run
import wandb


def run_epoch(
        model: MotifCaller, model_config: ModelConfig, optimizer: torch.optim, decoder: any, 
        X: List[torch.tensor], y: List[List[int]], ctc: CTCLoss, train: bool = False,
        display: bool = False, windows: bool = True) -> Dict[str, any]:

    n_training_samples = len(X)
    losses = np.zeros(n_training_samples)
    edit_distance_ratios = []
    greedy_transcripts = []
    display_iterations = int(n_training_samples / 2)
    device = model_config.device

    if train:
        model.train()
    else:
        model.eval()
    
    for ind in tqdm(range(n_training_samples)):
        
        if train:
            optimizer.zero_grad()

        if windows:
            input_sequence = X[ind].to(device)
        else:
            input_sequence = torch.tensor(
                X[ind], dtype=torch.float32)
            input_sequence = input_sequence.view(1, 1, len(X[ind])).to(device)

        target_sequence = torch.tensor(y[ind]).to(device)
        
        model_output = model(input_sequence)
        
        if windows:
            #print(model_output.shape)
            model_output = model_output.view(
                model_output.shape[0] * model_output.shape[1], model_config.n_classes)
            #print(model_output.shape)
        else:
            model_output = model_output.permute(1, 0, 2).view(
                model_output.shape[0] * model_output.shape[1], model_config.n_classes)
        
        n_timesteps = model_output.shape[0]
        input_lengths = torch.tensor(n_timesteps)
        label_lengths = torch.tensor(len(target_sequence))

        loss = ctc(
            model_output, target_sequence, input_lengths, label_lengths)
        
        if train:
            try:
                loss.backward()
            except Exception as e:
                print(f"Exception {e}")
                print(target_sequence)
                continue
            optimizer.step()
        
        losses[ind] = loss.item()
        
        if ind % 1000 == 0:
            greedy_result = decoder(model_output)
            greedy_transcript = " ".join(greedy_result)
            actual_transcript = get_actual_transcript(y[ind])
            edit_distance_ratios.append(ratio(greedy_transcript, actual_transcript))
            greedy_transcripts.append(greedy_transcript)
            
        if display:
            if ind % display_iterations == 0 and ind > 0:
                print(f"\nLoss is {loss.item()}")
                print(f"Ratio is {np.mean(edit_distance_ratios)}\n")
        
    return {
        "model": model,
        "losses": losses,
        "edit_distance_ratios": edit_distance_ratios,
        "greedy_transcripts": greedy_transcripts
    }
        

def main(
        epochs: int = 50, sampling_rate: float = 1.0, window_size: int = 1024,
        window_step: int = 800, n_classes:int = 10, running_on_hpc: bool = False,
        windows: bool = True):
    
    dataset_path, model_save_path, file_write_path = get_savepaths(
        running_on_hpc=running_on_hpc)

    X, y = load_training_data(
        dataset_path, column_x='squiggle', column_y='motif_seq',
        sampling_rate=sampling_rate)

    if windows:
        X = data_preproc(
            X=X, window_size=window_size, step_size=window_step, normalize_values=True)

    y = create_label_for_training(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=1) 
    torch.autograd.set_detect_anomaly(True)

    hidden_size = 256
    num_layers = 4
    output_size = n_classes
    dropout_rate = 0.2
    saved_model = False
    model_save_epochs = 5
    n_classes = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    print(f"Running on {device}")

    model = MotifCaller(n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


    labels_int = np.arange(n_classes).tolist()
    labels = [f"{i}" for i in labels_int] # Tokens to be fed into greedy decoder
    greedy_decoder = GreedyCTCDecoder(labels=labels)


    model_config = ModelConfig(
        n_classes=n_classes, hidden_size=hidden_size, window_size=window_size,
        window_step=window_step, train_epochs=epochs, device=device,
        model_save_path = model_save_path, write_path=file_write_path,
        dataset='synthetic', windows=windows
    )
    
    ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    if running_on_hpc:
        project_name = "motifcaller_hpc_runs"
    else:
        project_name = "motifcaller_local_runs"

    # Training monitoring on wandb
    wandb_login(running_on_hpc=running_on_hpc)
    start_wandb_run(model_config=model_config, project_name=project_name)

    for epoch in range(epochs):

        train_dict = run_epoch(
            model=model, model_config=model_config, optimizer=optimizer, decoder=greedy_decoder,
            X=X_train, y=y_train, ctc=ctc, train=True, display=False, windows=windows
        )

        model = train_dict['model']
        training_losses = train_dict['losses']
        training_ratios = train_dict['edit_distance_ratios']
        training_greedy_transcripts = train_dict['greedy_transcripts']
        
        print(f"\nTrain Epoch {epoch}\n Mean Loss {np.mean(training_losses)}"
              f"\n Mean ratio {np.mean(training_ratios)}\n")
        print(random.sample(training_greedy_transcripts, 1))
            
        validate_dict = run_epoch(
            model=model, model_config=model_config, optimizer=optimizer, decoder=greedy_decoder,
            X=X_val, y=y_val, ctc=ctc, display=False, windows=windows
            )
        
        validation_losses = validate_dict['losses']
        validation_ratios = validate_dict['edit_distance_ratios']
        validation_greedy_transcripts = train_dict['greedy_transcripts']

        metrics = {
            "train/accuracy": np.mean(training_ratios),
            "train/loss": np.mean(training_losses),
            "validation/accuracy": np.mean(validation_ratios),
            "validation/loss": np.mean(validation_losses)
        }

        wandb.log(metrics)

        # Schedule learning rate change
        scheduler.step(np.mean(validation_losses))
        current_lr = optimizer.param_groups[0]['lr']
        print(current_lr)

        print(f"\nValidation Epoch {epoch}\n Mean Loss {np.mean(validation_losses)}"
              f"\n Mean ratio {np.mean(validation_ratios)}\n")
        print(random.sample(validation_greedy_transcripts, 1))


        with open(file_write_path, 'a') as f:
            f.write(f"\nEpoch {epoch}\n Training loss {np.mean(training_losses)}\n"
                    f"Validation loss {np.mean(validation_losses)}")
            f.write(f"\nEdit distance ratio: Training {np.mean(training_ratios)}\n"
                    f"Validation {np.mean(validation_ratios)}")
            f.write(f"Transcripts: \n{random.sample(training_greedy_transcripts, 1)}\n"
                    f"{random.sample(validation_greedy_transcripts, 1)}\n")
        
        if epoch % model_save_epochs == 0 and epoch > 0:
            if model_save_path:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_save_path)
    
    test_dict = run_epoch(
        model=model, model_config=model_config, optimizer=optimizer, decoder=greedy_decoder,
        X=X_test, y=y_test, ctc=ctc, windows=windows
        )
    
    test_losses = validate_dict['losses']
    test_ratios = validate_dict['edit_distance_ratios']
    test_greedy_transcripts = train_dict['greedy_transcripts']
    print(f"\nTest Loop\n Mean Loss {np.mean(test_losses)}\n"
          f"Mean ratio {np.mean(test_ratios)}\n")

    with open(file_write_path, 'a') as f:
        f.write(f"\nTest loss {np.mean(test_losses)}\n Test ratio"
                f" {np.mean(test_ratios)}\nTest transcripts :\n"
                f"{random.sample(validation_greedy_transcripts, 1)}")

    if model_save_path:
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, model_save_path)