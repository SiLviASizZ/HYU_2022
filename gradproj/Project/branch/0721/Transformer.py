import tensorflow as tf
import numpy as np

# hyperparameter start

d_mass_int = 56
d_mass_float = 4
d_intensity = 4
num_heads = 8
d_model = d_mass_int + d_mass_float + d_intensity
dff = 128
num_peak = 500
rate = 0.1

# hyperparameter end

# positional encoding start

# same from basic transformer, need to switch pos_encoding function ?

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2*(i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

# positional encoding end


# Multi Head Attention start




# Multi Head Attention end


# Encoder Layer Start




# Encoder layer end








# Encoder start

def Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, num_peak, rate) # d_model 만 넘기고 나머지 3개는 Transformer 에서 임베딩해서 넣는게 어떤가?
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        # embedding : already done
        # self.pos_encoding = positional_encoding(self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:, :tf.shape(x)[1], :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training) # mask?
        
        return x


# Encoder end

# Decoder start

def Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, num_digestion, rate):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.pos_encoding = ?

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training):
        attention_weights = {}
        
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:, tf.shape(x)[1], :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training) # mask ?

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        
        return x, attention_weights

# Decoder end


# Transformer start

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_mass_int, d_mass_float, d_intensity, num_heads, dff, num_peak, num_digestion, rate):
        super().__init__()
        self.d_model = d_mass_int + d_mass_float + d_intensity

        self.encoder = Encoder(num_layers, self.d_model, num_heads, dff, num_peak, rate)
        self.decoder = Decoder(num_layers, self.d_model, num_heads, dff, num_digestion, rate)

        self.embedding_mass_int = tf.keras.layers.Embedding(1, d_mass_int)
        self.embedding_mass_float = tf.keras.layers.Embedding(1, d_mass_float)
        self.embedding_intensity = tf.keras.layers.Embedding(1, d_intensity)

        self.flnal_layer = tf.keras.layers.Dense(1) # squeeze 해서 2차원 출력할 거임, ( batch_size, num_peaks ) , value = probability of truth
    
    def call(self, input, target, training) # input : return value of mgfToTensor().call , target : return value of mgToTensor.get_sequence 

        # input : (batch_size, num_peak, 3)
        # target : (batch_size, num_digestion, 3)

        # masking ?
        enc_input = self.embedding(self, input)
        dec_input = self.embedding(self, target)

        enc_output = self.encoder(enc_input, training)
        dec_output = self.decoder(dec_input, enc_output, training)

        final_output = tf.squeeze(self.final_layer(dec_output))

        return final_output

    def embedding(self, input, d_mass_int, d_mass_float, d_intensity): # Transformer call 호출하면 input, target 를 임베딩 하고 시작하는게 어떤가?
        # input : ( batch_size, num_peak(or num_digestion), 3)
        # output : ( batch_size, num_peak(or num_digestion), d_model)
        embedding_matrix = tf.split(input, num_or_size_splits=3, axis=-1) # mass_int, mass_float, intensity 에 대해서 따로 Embedding 후 concatenate
        return tf.squeeze(tf.concat([self.embedding_mass_int(embedding_matrix[0]),
                                        self.embedding_mass_float(embedding_matrix[1]), 
                                        self.embedding_intensity(embedding_matrix[2])], axis = -1))



# Transformer end 