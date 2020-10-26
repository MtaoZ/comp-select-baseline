# Function to build our wave net model
def build_model(seq_len = 107, pred_len = 68, embed_dim = 85, dropout = 0.10):
    
    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2 ** i for i in range(n)]
        x = tf.keras.layers.Conv1D(filters = filters, 
                                   kernel_size = 1,
                                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = tf.keras.layers.Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = tf.keras.layers.Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
            x = tf.keras.layers.Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)
            res_x = tf.keras.layers.Add()([res_x, x])
        return res_x

    inputs = tf.keras.layers.Input(shape = (seq_len, 3))
    embed = tf.keras.layers.Embedding(input_dim = len(token2int), output_dim = embed_dim)(inputs)
    reshaped = tf.reshape(embed, shape = (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
    reshaped = tf.keras.layers.SpatialDropout1D(dropout)(reshaped)
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, 
                                                          dropout = dropout, 
                                                          return_sequences = True, 
                                                          kernel_initializer = 'orthogonal'))(reshaped)
    x = wave_block(x, 16, 3, 12)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = wave_block(x, 32, 3, 8)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = wave_block(x, 64, 3, 4)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    x = wave_block(x, 128, 3, 1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    
    truncated = x[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation = 'linear')(truncated)
    model = tf.keras.models.Model(inputs = inputs, outputs = out)
    opt = tf.keras.optimizers.Adam(learning_rate = LR)
    opt = tfa.optimizers.SWA(opt)
    model.compile(optimizer = opt,
                  loss = tf.keras.losses.MeanSquaredLogarithmicError(),
                  metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
    return model

