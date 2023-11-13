from hyperparamters import *


def get_embedder(
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        name='Embedder'):
    cover_im = Input(shape=input_shape)
    watermark = Input(shape=watermark_shape)

    cover_im_ = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)(cover_im)
    watermark_ = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size // 16, p2=patch_size // 16)(
        watermark)

    cover_im_ = vit_layers.AddPositionEmbs(name="T_Pos_Embed_c")(cover_im_)
    watermark_ = vit_layers.AddPositionEmbs(name="T_Pos_Embed_w")(watermark_)

    # MHA
    attention_output_1 = MultiHeadAttention(num_heads=num_heads, key_dim=mlp_dim)(cover_im_, watermark_)
    attention_output_2 = MultiHeadAttention(num_heads=num_heads, key_dim=mlp_dim)(watermark_, cover_im_)

    attention_output_1 = Add()([cover_im_, attention_output_1])
    attention_output_2 = Add()([watermark_, attention_output_2])

    attention_output = Concatenate()([attention_output_1, attention_output_2])

    x = Dense(768)(attention_output)
    h = 128 // patch_size

    marked_im = Rearrange('b (h w) (p1 p2 c) -> b (h p1) (w p2) c', h=h, w=h, p1=patch_size, p2=patch_size, c=3)(x)

    model = Model([cover_im, watermark], marked_im, name=name)

    return model