
import argparse
import os 
from datetime import datetime
from inference import model_init, model_inference
from output import save_inference_to_csv


parser = argparse.ArgumentParser(
                    prog='Motif Caller',
                    description='Inference for the motif caller')

parser.add_argument('--fast5_path', type=str)
parser.add_argument('--savepath', type=str)

parser.set_defaults(
    fast5_path="",
    savepath=""
    )

args = parser.parse_args()

if __name__ == '__main__':
    fast5_path = args.fast5_path
    savepath = args.savepath

    if not savepath:
        print("No savepath provided, saving output to current directory\n")

    if fast5_path:
        print("Initialising model and reading data\n")
        squiggles, read_ids, model, device, greedy_decoder = model_init(fast5_path=fast5_path)
        print("Starting inference\n")
        sorted_greedy_transcripts, greedy_transcripts_arr, read_ids_arr = model_inference(
            data_arr=squiggles, read_ids=read_ids, model=model, device=device, greedy_decoder=greedy_decoder
        )
        print("Saving results\n")
        save_inference_to_csv(sorted_greedy_transcripts=sorted_greedy_transcripts, greedy_transcripts_arr=greedy_transcripts_arr, read_ids_arr=read_ids_arr, savepath=savepath, fast5_filepath=fast5_path)

    else:
        print("No fast5 path provided!")
        exit()