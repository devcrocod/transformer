import torch
from trans import Transformer
from text_util import TextEncoder


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    text_encoder = TextEncoder("encoder_bpe_40000.bpe", "vocav_40000.bpe")
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)



    args = DEFAULT_CFG
    model = Transformer(args)



class ArgDict(dict):
    """for attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DEFAULT_CFG = ArgDict({
    'n_embd': 512,
    'hn_head': 12,
    'n_layer': 12,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'clf_pdrop': 0.1
})
