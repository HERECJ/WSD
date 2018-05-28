from keras.layers import Layer
import keras.backend as K


class ConcatCon(Layer):
    def __init__(self,**kwargs):
        self.supports_masking = True
        super(ConcatCon, self).__init__(**kwargs)

    def build(self, input_shape):
        #the input of the layer is output_sequences of lstm, and the attention output
        #in another word it is a list
        #the shape of output_lstm is (batch_size,max_len,hidden_size)
        #the shape of attention output is (batch_size,hidden_size)
        self.hidden_size = input_shape[0][-1]
        self.batch_size = input_shape[0][0]
        self.timestep = input_shape[0][1]
        super(ConcatCon, self).build(input_shape)

    def call(self, inputs, mask=None):
        lstm_output = inputs[0]
        attention_output = inputs[1]
        attention_tensor = K.repeat(attention_output,self.timestep)
        mask = mask[0]
        mask = K.expand_dims(mask,axis=-1)
        mask = K.tile(mask,[1,1,self.hidden_size])
        if mask is not None:
            attention_tensor *= K.cast(mask,dtype='float32')
        return K.concatenate([lstm_output,attention_tensor],-1)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        timestep = input_shape[0][1]
        hidden_size = input_shape[0][-1] * 2
        batch_size = input_shape[0][0]
        return  (batch_size,timestep,hidden_size)

