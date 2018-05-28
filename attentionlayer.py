from keras.layers import Layer
import keras.backend as K


class Attention(Layer):
    def __init__(self,**kwargs):
        """
        Attention operation for temporal data.
        # Input shape
            3D tensor with shape: `(batch_size, input_length, hidden_size)`.
        # Output shape
            2D tensor with shape: `(batch_size, hidden_size)`.
        :param kwargs:
        """
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # the input shape (batch_size,max_length,hidden_size*2)
        # *2 means the bidirectional lstm
        # create the trainable weight variable for this layer

        # input_shape[-1] = hidden_size
        self.att_size = input_shape[-1]
        self.batch_size = input_shape[0]
        self.timestep = input_shape[1]
        self.Wq = self.add_weight(name='parameter_vector', shape=(self.att_size,1), initializer='glorot_normal',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs, mask=None):
        memory = inputs
        hidden = K.squeeze(K.dot(memory,self.Wq),axis=-1)
        hidden = K.tanh(hidden)
        s = K.softmax(hidden)
        if mask is not None:
            s *= K.cast(mask,dtype='float32')
            sum_by_time = K.sum(s,axis=-1,keepdims=True)
            s = s/(sum_by_time + K.epsilon())
        return K.sum(memory * K.expand_dims(s),axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        att_size = input_shape[-1]
        batch_size = input_shape[0]
        return  (batch_size,att_size)

