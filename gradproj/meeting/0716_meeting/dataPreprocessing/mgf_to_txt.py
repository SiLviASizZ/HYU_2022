input_file_path = "./db\human_peaks_db_sample.mgf"
target_file_path = "./data\data.txt"

output=open(target_file_path, 'w+')
f=open(input_file_path, 'r')

batch_db_dict = {}
SEQ = ''
PEPMASS = 0.
CHARGE = 0.  
num_peak = 0
first_dict = {'1.007276035': 1.0100, '19.017840735': 1.0200}

def normalize(max, v):
    return v/max

def find_end_value(p, c):
    y = (p-1.007276035)*CHARGE
    return y, y-18.0105647

while True:
    line = f.readline().replace("\n","\t")
    if not line:
        break
    if line == 'BEGIN IONS\t':
        while(True):
            batch_line = f.readline().replace("\n","\t").replace("="," ")
            if batch_line == 'END IONS\t':
                # batch_db_dict
                num_peaks = len(batch_db_dict) + 4
                batch_max = max((batch_db_dict).values())
                first, second = find_end_value(PEPMASS, CHARGE)

                output.write(f"{num_peaks}\t")
                for x, y in first_dict.items():
                    output.write(f"{x}\t{y}\t")
                for x, y in batch_db_dict.items():
                    output.write(f"{x}\t{normalize(batch_max, float(y))}\t")
                output.write(f"{first}\t1.0100\t{second}\t1.0200\t")
                output.write(f"{SEQ}\t{CHARGE}\t{PEPMASS}\n")
                batch_db_dict.clear()
                break
            else:
                (first, second) = batch_line.split()
                if first == 'PEPMASS':
                    PEPMASS = float(second.strip())
                    # print(PEPMASS)
                elif first == 'CHARGE':
                    CHARGE = float(second.strip('+-'))
                    # print(CHARGE)
                elif first == 'SEQ':
                    SEQ = (second.strip().replace("C(+57.02)","C").replace("M(+15.99)","m").replace("N(+.98)","n")
                                        .replace("Q(+.98)","q").replace("S(+79.97)","s")
                                        .replace("T(+79.97","t").replace("Y(+79.97)","y"))
                    # print(SEQ)
                elif (first == 'TITLE') or (first == 'SCANS') or (first == 'RTINSECONDS'):
                    continue
                else: # peak, batch_db_dict is dict of peaks, (position, intensity)
                    batch_db_dict[first] = float(second) # key:string, value:float
    # print("batch over")