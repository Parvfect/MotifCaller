
import pysam
import pandas as pd


bam_filepath = "calls.bam"

samfile = pysam.AlignmentFile(bam_filepath, "rb", check_sq=False)

read_ids = []
base_seq = []

for s in samfile:
    read_ids.append(str(s).split('!')[1][1:])
    base_seq.append(str(s.seq))    

samfile.close()


df = pd.DataFrame({"read_id": read_ids, "base_seq": base_seq})

print(df.head())

df.to_csv("foward_basecalled.csv")