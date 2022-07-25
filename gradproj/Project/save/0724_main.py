import tensorflow as tf
import numpy as np

class MgfToTensor():
    def __init__(self):
        self.first_matrix = [[1, 0, 101], [19, 1, 102]]
        self.return_matrix = []
        self.max_length = 0
        self.num_spectrum = 0
        self.return_tensor = tf.zeros(0)
        self.SEQ_list = []
        self.CHARGE_list = []

    def call(self, file_path):
        f = open(file_path, 'r')
        self.num_spectrum = 0
        self.max_length = 0
        self.return_matrix.clear()
        self.return_tensor = tf.zeros(0)
        self.SEQ_list.clear()
        self.CHARGE_list.clear()

        while True:
            batch_matrix = []
            PEPMASS = 0
            CHARGE = 0
            max_value = 0.
            SEQ = ''

            line = f.readline()
            if not line:
                break
            if line == 'BEGIN IONS\n':
                self.num_spectrum += 1
                while(True):
                    batch_line = f.readline().replace("="," ")
                    if batch_line == 'END IONS\n':
                        x = (PEPMASS-1.007276035)*CHARGE + 1.007276035
                        y = x - 18.0105647

                        for i in range(len(batch_matrix)):
                            batch_matrix[i][2] /= max_value
                            batch_matrix[i][2] = int(batch_matrix[i][2]*100)
                        batch_matrix.insert(0, self.first_matrix[1])
                        batch_matrix.insert(0, self.first_matrix[0])
                        batch_matrix.insert(0, [0, 0, 0])
                        batch_matrix.insert(0, [0, 0, CHARGE])

                        batch_matrix.append(self.get_batch(y, 101)) # y : b-ion
                        batch_matrix.append(self.get_batch(x, 102)) # x : y-ion
                        

                        self.return_matrix.append(batch_matrix)


                        if len(batch_matrix) > self.max_length:
                            self.max_length = len(batch_matrix)
                        break
                    else:
                        (first, second) = batch_line.split()
                        if first == 'PEPMASS':
                            PEPMASS = float(second.strip())
                        elif first == 'CHARGE':
                            CHARGE = float(second.strip('+-'))
                            self.CHARGE_list.append(CHARGE)
                        elif first == 'SEQ':
                            SEQ = (second.strip().replace("C(+57.02)","C").replace("M(+15.99)","m").replace("N(+.98)","n")
                                        .replace("Q(+.98)","q").replace("S(+79.97)","s")
                                        .replace("T(+79.97","t").replace("Y(+79.97)","y"))
                            self.SEQ_list.append(SEQ)
                        elif (first == 'TITLE') or (first == 'SCANS') or (first == 'RTINSECONDS'):
                            continue
                        else: # peak
                            m_over_z = float(first)
                            intensity = float(second)
                            if max_value < intensity:
                                max_value = intensity
                            batch_matrix.append(self.get_batch(m_over_z, intensity))
        f.close()
        self.return_tensor = tf.ragged.constant(self.return_matrix).to_tensor(default_value=0)
        return self.return_tensor # (num_spectrum, max_length, 3)

    def get_batch(self, first, second): # first, second are strings, e.g. '174.246', '24.19882'
        x = int(first)
        y = first - x
        return [x, int(y*100), second]

    def get_sequence(self):
        return self.SEQ_list

    def get_charge(self):
        return self.CHARGE_list


transform = MgfToTensor()
k = transform.call("./db\human_peaks_db_sample.mgf")

sequence = transform.get_sequence() # 1 dimensional array, ['seq1', 'seq2' ... ]
charge = transform.get_charge() # 1 dimensional array, [1., 4., 2., 3. ...]

def tokenize_seq(seq_input, charge_input):
    # get string ( sequence ) as input
    # output is 2-dimensional array, [num_digestion, 3]
    # num_digestion includes the whole sequence for b-ion and y-ion respectively
    
    
    # Tokenize table Begin
     
    Proton = 1.007276035
    H2O = 18.0105647

    amino2idx = {'<pad>': 0, '<bos>': 1, '<eos>': 2,
             'A': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7,
             'G': 8, 'H': 9, 'I': 10, 'K': 11, 'L': 12,
             'M': 13, 'N': 14, 'P': 15, 'Q': 16, 'R': 17,
             'S': 18, 'T': 19, 'V': 20, 'W': 21, 'Y': 22,
             'm': 23, 'n': 24, 'q': 25, 's': 26, 't': 27, 'y': 28}

    idx2amino = {0: '<pad>',1: '<bos>', 2: '<eos>',
             3: 'A', 4: 'C', 5: 'D', 6: 'E', 7: 'F',
             8: 'G', 9: 'H', 10: 'I', 11: 'K', 12: 'L',
             13: 'M', 14: 'N', 15: 'P', 16: 'Q', 17: 'R',
             18: 'S', 19: 'T', 20: 'V', 21: 'W', 22: 'Y',
             23: 'm', 24: 'n', 25: 'q', 26: 's', 27: 't', 28: 'y'}

    idx2mass = {0: 0.00, 1: 0.00, 2: 0.00,
             3: 71.03, 4: 160.03, 5: 115.02, 6: 129.04, 7: 147.06,
             8: 57.021, 9: 137.05, 10: 113.08, 11: 128.09, 12: 113.08,
             13: 131.04, 14: 114.04, 15: 97.05, 16: 128.05, 17: 156.10,
             18: 87.03, 19: 101.04, 20: 99.06, 21: 186.07, 22: 163.06,
             23: 147.03, 24: 115.02, 25: 129.04, 26: 167.00, 27: 181.02, 28:243.04}

    # Tokenize table End

    b_ion_seq = list(seq_input)
    y_ion_seq = list(reversed(b_ion_seq))
    b_ion_tokenize = []
    y_ion_tokenize = []
    b_ion_mass = []
    y_ion_mass = []
    concatenated_mass = []
    result_matrix = []

    for i in range(len(b_ion_seq)):        ## transform to mass through given dictionary
        # amino -> index
        b_ion_tokenize.append(amino2idx[b_ion_seq[i]]) # ion_tokenize : index
        y_ion_tokenize.append(amino2idx[y_ion_seq[i]])
        # index -> mass
        b_ion_mass.append(idx2mass[b_ion_tokenize[i]]) # ion_mass : mass of each characters
        y_ion_mass.append(idx2mass[y_ion_tokenize[i]])
    b_ion_mass = np.cumsum(b_ion_mass) + 1.007276035 # ion_mass : mass of each digestion
    y_ion_mass = np.cumsum(y_ion_mass) + 19.017840735
    
    concatenated_mass = np.sort(np.concatenate([b_ion_mass, y_ion_mass]))
    
    result_matrix.append([0, 0, charge_input])
    result_matrix.append([0, 0, 0])
    for i in range(len(concatenated_mass)):
        result_matrix.append([int(concatenated_mass[i]), int((float(concatenated_mass[i])-int(concatenated_mass[i]))*100), 50])

    return result_matrix

for i in range(len(sequence)):
    print(tf.cast(tokenize_seq(sequence[i], charge[i]), dtype=tf.int64))

