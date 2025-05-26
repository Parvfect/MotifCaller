
from ont_fast5_api.fast5_interface import get_fast5_file
import pandas as pd


def get_data_from_fast5(fast5_filepath: str):
    raw_data_arr = []
    read_ids = []
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        for read in f5.get_reads():
            raw_data = read.get_raw_data()
            raw_data_arr.append(raw_data)
            read_ids.append(read.read_id)
    return raw_data_arr, read_ids


def get_cleaned_encoded_file(encoded_df):

    # Joining payloads
    payload_cols = [col for col in encoded_df.columns if col.startswith('Payload')]
    encoded_df['payload'] = encoded_df[payload_cols].astype(str).agg(', '.join, axis=1)

    # Fixing addresses
    encoded_df['Address_Incrementer_1'] = encoded_df['Address_Incrementer_1'].apply(lambda x: f'barcode_external0{x[1]}')
    encoded_df['Address_Incrementer_2'] = encoded_df['Address_Incrementer_2'].apply(lambda x: f'_internal0{x[1]}')
    address_cols = [col for col in encoded_df.columns if col.startswith('Address')]
    encoded_df['HW_Address'] = encoded_df[address_cols].astype(str).agg(''.join, axis=1)

    # Selecting important columns
    encoded_df = encoded_df[['HW_Address', 'payload']]
    return encoded_df