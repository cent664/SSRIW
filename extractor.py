from hyperparamters import *


def get_extractor(
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        name='Extractor'):
    marked_im = Input(shape=input_shape)

    x = Reshape((8, 8, 768))(marked_im)

    x = Conv2D(64, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = Activation('relu')(x)

    x = Dense(512)(x)

    x = Conv2D(128, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(8, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(1, 3, padding='same')(x)
    watermark = Activation('relu')(x)

    model = Model(marked_im, watermark, name=name)

    return model