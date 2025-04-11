from nn import CallerEmpirical
import torch
from decoder import GreedyCTCDecoder
import os
from tqdm import tqdm
from sklearn.preprocessing import normalize
from torch.nn.utils.rnn import pad_sequence
from utils import sort_transcript
import pandas as pd
from typing import List
from fast5_input import extract_fast5_data_from_file
from typing import List, Tuple


n_classes = 19
hidden_size = 256

def model_init(fast5_path: str):

    squiggles, read_ids = extract_fast5_data_from_file(fast5_filepath=fast5_path)
    model_path = 'model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(f"Running on {device}")

    # Initialising decoder
    greedy_decoder = GreedyCTCDecoder(n_classes=19)

    model = CallerEmpirical(num_classes=n_classes, hidden_dim=hidden_size)
        
    # Loading model from checkpoint
    if device == torch.device('cpu'):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path)
        
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    return squiggles, read_ids, model, device, greedy_decoder

def model_inference(
        data_arr: List[List[int]], read_ids: List[str],
        model: torch.nn, device: torch.device,
        greedy_decoder: GreedyCTCDecoder) -> Tuple[List[List[int]], List[str], List[str]]:

    greedy_transcripts_arr = []
    sorted_greedy_transcripts = []
    read_ids_arr = []

    n_training_samples = len(data_arr)

    print(f"Inference on {n_training_samples} squiggles")

    batch_size = 8
    model = model.to(device)

    with torch.no_grad():
        for ind in tqdm(range(0, n_training_samples, batch_size)):

            if n_training_samples - ind < batch_size:
                # Add random seqs to the end and get an output still
                continue
            
            input_seqs = [
                normalize([data_arr[k]], norm='max').flatten() for k in range(ind, ind + batch_size)]
            

            input_seqs = pad_sequence([torch.tensor(
                        i, dtype=torch.float32) for i in input_seqs], batch_first=True)
            
            input_seqs = input_seqs.view(input_seqs.shape[0], 1, input_seqs.shape[1])
            input_seqs = input_seqs.to(device)

            try:
                model_output = model(input_seqs)
                if device.type == 'cuda':
                    model_output = model_output.cpu()
                
                for k in range(batch_size):
                    greedy_result = greedy_decoder(model_output[k])
                    greedy_transcript = " ".join(greedy_result)
                    sorted_greedy = sort_transcript(greedy_transcript)
                    greedy_transcripts_arr.append(greedy_transcript)
                    sorted_greedy_transcripts.append(sorted_greedy)
                torch.cuda.empty_cache()
                
                read_ids_arr.extend(read_ids[ind: ind + batch_size])

            except Exception as e:
                print("Ignoring error {e} and continuing inference")
                continue

    return sorted_greedy_transcripts, greedy_transcripts_arr, read_ids_arr