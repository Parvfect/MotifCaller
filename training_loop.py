

# fuck jupyter
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


n_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
print(f"Running on {device}")

model = MotifCaller(output_classes=n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
labels_int = np.arange(n_classes).tolist()
labels = [f"{i}" for i in labels_int] # Tokens to be fed into greedy decoder
greedy_decoder = GreedyCTCDecoder(labels=labels)

def run_epoch(
        model: MotifCaller, X: List[torch.tensor], y: List[List[int]],
        ctc: CTCLoss, model_save_path: str = "", model_write_path: str = "",
        save_iterations: int = 1000, train: bool = False, 
        display: bool = False) -> Dict[str, any]:

    n_training_samples = len(X)
    losses = np.zeros(n_training_samples)
    edit_distance_ratios = np.zeros(n_training_samples)
    greedy_transcripts = []
    display_iterations = int(n_training_samples / 2)

    if train:
        model.train()
    else:
        model.eval()
    
    for ind in tqdm(range(n_training_samples)):
        
        input_sequence = X[ind].to(device)
        target_sequence = torch.tensor(y[ind]).to(device)
        
        model_output = model(input_sequence)

        model_output = model_output.view(
            model_output.shape[0] * model_output.shape[1], n_classes)
        
        n_timesteps = model_output.shape[0]
        input_lengths = torch.tensor(n_timesteps)
        label_lengths = torch.tensor(len(target_sequence))

        loss = ctc(
            model_output, target_sequence, input_lengths, label_lengths)
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        greedy_result = greedy_decoder(model_output)
        greedy_transcript = " ".join(greedy_result)
        actual_transcript = get_actual_transcript(y[ind])
        edit_distance_ratio = ratio(greedy_transcript, actual_transcript)

        losses[ind] = loss.item()
        edit_distance_ratios[ind] = ratio(greedy_transcript, actual_transcript)
        greedy_transcripts.append(greedy_transcript)

        if display:
            if ind % display_iterations == 0 and ind > 0:
                print(f"\nLoss is {loss.item()}")
                print(f"Ratio is {edit_distance_ratio}\n")
        
        
    return {
        "model": model,
        "losses": losses,
        "edit_distance_ratios": edit_distance_ratios,
        "greedy_transcripts": greedy_transcripts
    }
        
        
if __name__ == "__main__":
    
    dataset_path, model_save_path, file_write_path = get_savepaths(
        running_on_hpc=True)
    
    #df = pd.read_pickle(dataset_path)
    X, y = load_training_data(
        dataset_path, column_x='Squiggle', column_y='Motifs', sampling_rate=0.1)

    X = data_preproc(
        X=X, window_size=1024, step_size=800, normalize_values=True)
    y = create_label_for_training(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1) 
    torch.autograd.set_detect_anomaly(True)

    # Model parameters
    epochs = 35
    hidden_size = 256
    num_layers = 4
    output_size = n_classes
    dropout_rate = 0.2
    saved_model = False
    model_save_epochs = 5

    for epoch in range(epochs):

        train_dict = run_epoch(
            model=model, X=X_train, y=y_train, ctc=ctc,
            train=True, display=False
        )

        model = train_dict['model']
        training_losses = train_dict['losses']
        training_ratios = train_dict['edit_distance_ratios']
        training_greedy_transcripts = train_dict['greedy_transcripts']

        print(f"\nTrain Epoch {epoch}\n Mean Loss {np.mean(training_losses)}"
              f"\n Mean ratio {np.mean(training_ratios)}\n")
        print(random.sample(training_greedy_transcripts, 3))
            
        validate_dict = run_epoch(model=model, X=X_val, y=y_val, ctc=ctc,
                                    display=False)
        
        validation_losses = validate_dict['losses']
        validation_ratios = validate_dict['edit_distance_ratios']
        validation_greedy_transcripts = train_dict['greedy_transcripts']

        print(f"\nValidation Epoch {epoch}\n Mean Loss {np.mean(validation_losses)}"
              f"\n Mean ratio {np.mean(validation_ratios)}\n")
        print(random.sample(validation_greedy_transcripts, 3))


        with open(file_write_path, 'a') as f:
            f.write(f"\nEpoch {epoch} Training loss {np.mean(training_losses)}"
                    f"Validation loss {np.mean(validation_losses)}")
            f.write(f"\n Edit distance ratio: Training {np.mean(training_ratios)}"
                    f"Validation {np.mean(validation_ratios)}")
            f.write(f"Transcripts: {random.sample(training_greedy_transcripts, 1)}"
                    f"{random.sample(validation_greedy_transcripts, 1)}\n")
        
        if epoch % model_save_epochs == 0 and epoch > 0:
            if model_save_path:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_save_path)
    
    test_dict = run_epoch(model=model, X=X_test, y=y_test, ctc=ctc)
    test_losses = validate_dict['losses']
    test_ratios = validate_dict['edit_distance_ratios']
    print(f"\nTest Loop\n Mean Loss {np.mean(test_losses)}\n"
          f"Mean ratio {np.mean(test_ratios)}\n")

    with open(file_write_path, 'a') as f:
        f.write(f"\nTest loss {np.mean(test_losses)} Test ratio"
                f"{np.mean(test_ratios)}")

    if model_save_path:
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, model_save_path)