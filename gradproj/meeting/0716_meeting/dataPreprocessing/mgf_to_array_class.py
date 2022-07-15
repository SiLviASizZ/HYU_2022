import numpy as np
from pathlib import Path

class mgf_to_array():
    def __init__(self, file_path):
        self.file_path = file_path
        self.first_matrix = [[1, 0, 101], [19, 1, 102]]
        self.SEQ = ''
        self.PEPMASS = 0.
        self.CHARGE = 0.

        self.return_matrix = np.empty(shape=0)

    def call(self):
        batch_matrix = []
        f = open(self.file_path, 'r')
        while True:
            line = f.readline().replace("\n","\t")
            if not line:
                break
            if line == 'BEGIN IONS\t':
                while(True):
                    batch_line = f.readline().replace("\n","\t").replace("="," ")
                    if batch_line == 'END IONS\t':
                        print(batch_matrix)
                        self.return_matrix.append(np.array(batch_matrix))
                        batch_matrix.clear()
                        break
                    else:
                        (first, second) = batch_line.split()
                        if first == 'PEPMASS':
                            self.PEPMASS = float(second.strip())
                        elif first == 'CHARGE':
                            self.CHARGE = float(second.strip('+-'))
                        elif first == 'SEQ':
                            self.SEQ = (second.strip().replace("C(+57.02)","C").replace("M(+15.99)","m").replace("N(+.98)","n")
                                        .replace("Q(+.98)","q").replace("S(+79.97)","s")
                                        .replace("T(+79.97","t").replace("Y(+79.97)","y"))
                        elif (first == 'TITLE') or (first == 'SCANS') or (first == 'RTINSECONDS'):
                            continue
                        else: #peak
                            batch_matrix.append([first, first, second])
        f.close()

    def normalize(self, max, v):
        return v/max

    def find_end_value(self, p, c):
        y = (p-1.007276035)*CHARGE
        return y, y-18.0105647

    def check(self):
        print(self.return_matrix)


# for test
transform = mgf_to_array(Path("./db\human_peaks_db_sample.mgf"))
transform
transform.check()