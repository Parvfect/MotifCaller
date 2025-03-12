
import pysam
import pandas as pd
import json


filepath = r"calls.bam"
samfile = pysam.AlignmentFile(filepath, "rb", check_sq=False)
print(dir(samfile))

seq_id = []
positions = []

iter = samfile.fetch(until_eof=True)
for ind, x in enumerate(iter):
    print(str(x).split()[0])

    if ind == 10:
        break
    


with open(r"C:\Users\Parv\Doc\HelixWorks\Basecalling\code\motifcaller\data\synthetic\synthetic_dataset_preprocess_runs\25.2.25\info.json", 'r') as f:
    info_dict = json.load(f)