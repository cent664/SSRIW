from hyperparamters import *


def get_decoder(
        input_shape,
        num_layers=num_layers,
        patch_size=patch_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        dropout=.02,
        num_heads=num_heads,
        name='Decoder'
):
    in_channels = 3
    patch_dim = in_channels * patch_size ** 2
    h = 128 // patch_size

    ip = Input(shape=input_shape)
    y = ip

    for n in range(4):
        y, _ = vit_layers.TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"T_Dec_Block_{n}"
        )(y)

    y = LayerNormalization(
        epsilon=1e-6, name="T_LNorm"
    )(y)

    y = Dense(patch_dim)(y)
    y = Rearrange('b (h w) (p1 p2 c) -> b (h p1) (w p2) c', h=h, w=h, p1=patch_size, p2=patch_size, c=in_channels)(y)

    model = Model(inputs=ip, outputs=y, name=name)
    return model