import os
os.getcwd()
input_file_path = "./db\human_peaks_db_sample.mgf"
target_file_path = "./data\data.txt"

output=open(target_file_path, 'w+')

with open(input_file_path) as f:
    lines = f.readlines()
    output.write(str(lines))
