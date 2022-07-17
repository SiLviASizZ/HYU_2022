import pprint
import tensorflow as tf

class MgfToTensor():
    def __init__(self):
        self.first_matrix = [[1, 0, 1], [19, 1, 1]]
        self.return_matrix = []
        self.max_length = 0
        self.num_batch = 0
        self.return_tensor = tf.zeros(0)

    def call(self, file_path):
        f = open(file_path, 'r')
        self.num_batch = 0
        self.max_length = 0
        self.return_matrix = []
        self.return_tensor = tf.zeros(0)
        # max_value = 0

        while True:
            batch_matrix = []
            PEPMASS = 0
            CHARGE = 0 
            SEQ = ''

            line = f.readline().replace("\n","\t")
            if not line:
                break
            if line == 'BEGIN IONS\t':
                batch_matrix.append(self.first_matrix[0])
                batch_matrix.append(self.first_matrix[1])
                self.num_batch += 1
                while(True):
                    batch_line = f.readline().replace("\n","\t").replace("="," ")
                    if batch_line == 'END IONS\t':
                        
                        x = (PEPMASS-1.007276035)*CHARGE
                        y = x - 17.003288665
                        batch_matrix.append(self.get_batch(x, 1))
                        batch_matrix.append(self.get_batch(y, 1))
                        # for x in range(len(batch_matrix)):
                        #     self.return_matrix.append(batch_matrix[x])
                        self.return_matrix.append(batch_matrix)
                        if len(batch_matrix) > self.max_length:
                            self.max_length = len(batch_matrix)
                        break
                    else:
                        (first, second) = batch_line.split()
                        if first == 'PEPMASS':
                            PEPMASS = float(second.strip())
                        elif first == 'CHARGE':
                            CHARGE = float(second.strip('+-')) # is CHARGE always positive ?
                        elif first == 'SEQ':
                            SEQ = (second.strip().replace("C(+57.02)","C").replace("M(+15.99)","m").replace("N(+.98)","n")
                                        .replace("Q(+.98)","q").replace("S(+79.97)","s")
                                        .replace("T(+79.97","t").replace("Y(+79.97)","y"))
                        elif (first == 'TITLE') or (first == 'SCANS') or (first == 'RTINSECONDS'):
                            continue
                        else: #peak
                            if max_value < second:
                                max_value = second
                            batch_matrix.append(self.get_batch(first, second))
        f.close()
        # tf.convert_to_tensor(self.return_matrix, dtype=tf.float32)
        self.return_tensor = tf.ragged.constant(self.return_matrix).to_tensor(default_value=0)
        return self.return_tensor # (batch_num, max_length, 3)

    def get_batch(self, first, second):
        x = int(float(first))
        y = float(first) - x
        z = int(float(second))
        return [x, int(y*100), z]

    def check_length(self):
        print(self.max_length)

    def check_size(self):
        print(self.num_batch)



# for test
transform = MgfToTensor()
k = transform.call("./db\human_peaks_db_sample.mgf")
# transform.check()
pprint.pprint(k)

transform.check_length()
transform.check_size()
print(tf.shape(k))

# 1. charge : is it always positive ?
# 2. intensity output?