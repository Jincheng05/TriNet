import ml_collections

def get_trinet_large_config():
    config = ml_collections.ConfigDict()
    config.if_skip = True
    config.if_Dyt = True
    config.embed_dim = 32
    config.hidden_dim = 48
    config.inshape = (160, 192, 224)
    config.num_heads = 2
    config.qkv_bias = False
    return config

def get_trinet_config():
    config = ml_collections.ConfigDict()
    config.if_skip = True
    config.if_Dyt = True
    config.embed_dim = 16
    config.hidden_dim = 16
    config.inshape = (160, 192, 224)
    config.num_heads = 2
    config.qkv_bias = False
    return config

def get_trinet_small_config():
    config = ml_collections.ConfigDict()
    config.if_skip = True
    config.if_Dyt = True
    config.embed_dim = 8
    config.hidden_dim = 16
    config.inshape = (160, 192, 224)
    config.num_heads = 1
    config.qkv_bias = False
    return config