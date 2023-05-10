def get_model(num_class):
    import tensorflow as tf
    from keras.applications.xception import Xception, preprocess_input
    IMG_SHAPE = (224, 224, 3)

    base_model = Xception(
        input_shape = IMG_SHAPE,
        include_top = False,
        weights     = 'imagenet')
    
    base_model.trainable = False

    ip_layer = tf.keras.Input(shape=IMG_SHAPE)
    x        = preprocess_input(ip_layer)
    x        = base_model(x, training=False)
    x        = tf.keras.layers.GlobalAveragePooling2D()(x)
    x        = tf.keras.layers.Dense(512, activation='relu')(x)
    x        = tf.keras.layers.Dropout(.2)(x)
    op_layer = tf.keras.layers.Dense(num_class, activation='softmax')(x)

    model = tf.keras.Model(inputs=ip_layer, outputs=op_layer)
    
    return model