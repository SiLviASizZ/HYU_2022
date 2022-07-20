import pprint
import tensorflow as tf

class MgfToTensor():
    def __init__(self):
        self.first_matrix = [[1, 0, 1.0100], [19, 1, 1.0200]]
        self.return_matrix = []
        self.max_length = 0
        self.num_spectrum = 0
        self.return_tensor = tf.zeros(0)

    def call(self, file_path):
        f = open(file_path, 'r')
        self.num_spectrum = 0
        self.max_length = 0
        self.return_matrix = []
        self.return_tensor = tf.zeros(0)
        # max_value = 0

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
                        y = x - 17.003288665

                        for i in range(len(batch_matrix)):
                            batch_matrix[i][2] /= max_value
                        batch_matrix.insert(0, self.first_matrix[1])
                        batch_matrix.insert(0, self.first_matrix[0])

                        batch_matrix.append(self.get_batch(x, 1.0100))
                        batch_matrix.append(self.get_batch(y, 1.0200))
                        
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


transform = MgfToTensor()
k = transform.call("./db\human_peaks_db_sample.mgf")
# transform.check()
pprint.pprint(k)

transform.check_length()
transform.check_size()
print(tf.shape(k))
