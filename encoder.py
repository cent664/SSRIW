from hyperparamters import *


def get_encoder(
        num_layers=num_layers,
        patch_size=patch_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        dropout=.02,
        num_heads=num_heads,
        name='Encoder'
):
    in_channels = 3
    patch_dim = in_channels * patch_size ** 2
    h = 128 // patch_size

    ip = Input(shape=(128, 128, in_channels))

    y = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)(ip)

    y = Dense(hidden_size)(y)
    y = vit_layers.AddPositionEmbs(name="T_Pos_Embed")(y)

    for n in range(num_layers):
        y, _ = vit_layers.TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"T_Enc_Block_{n}"
        )(y)

    y_ = BatchNormalization()(y)
    y_ = Flatten()(y_)
    y_ = Dense(1000)(y_)

    model = Model(inputs=ip, outputs=[y, y_], name=name)
    return model