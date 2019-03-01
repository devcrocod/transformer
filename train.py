import os
import random
import time
import json
import csv

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from data_utils import Corpus
from loss import MultipleChoiceLossCompute
from text_util import TextEncoder
from trans import Model
from utils import encode, iter_data, make_path

n_embd = 768
n_head = 12
n_layer = 12
embd_pdrop = 0.1
attn_pdrop = 0.1
resid_pdrop = 0.1
clf_pdrop = 0.1


def transform_roc(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [clf_token]
        x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        l13 = len(x13)
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb


def _stories(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1]) - 1)
        return st, ct1, ct2, y


def stories(data_dir, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _stories(os.path.join(data_dir, 'cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _stories(os.path.join(data_dir, 'cloze_test_ALL_test.csv'))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1,
                                                                                                      comps2, ys,
                                                                                                      test_size=n_valid,
                                                                                                      random_state=3535999445)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)


def iter_apply(Xs, Ms, Ys):
    logits = []
    cost = 0
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            clf_logits *= n
            clf_losses = compute_loss_fct(XMB, YMB, MMB, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost


def iter_predict(Xs, Ms):
    logits = []
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            logits.append(clf_logits.to("cpu").numpy())
    logits = np.concatenate(logits, 0)
    return logits


def log(save_dir, desc):
    global best_score
    print("Logging")
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_cost / len(trY[:n_valid])
    va_cost = va_cost / n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, desc, 'best_params')
            torch.save(dh_model.state_dict(), make_path(path))


def predict(dataset, submission_dir):
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    predictions = pred_fn(iter_predict(teX, teM))

    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))


def run_epoch():
    for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trYt, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, clf_logits = dh_model(XMB)
        compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)
        n_updates += 1
        if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
            log(save_dir, desc)


class ResultLogger(object):
    def __init__(self, path, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log = open(make_path(path), 'w')
        self.f_log.write(json.dumps(kwargs) + '\n')

    def log(self, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log.write(json.dumps(kwargs) + '\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()


def get_corpus(datadir, dataset):
    fn = os.path.join(datadir, 'cache.pt')
    if os.path.exists(fn):
        print('Load cached dataset...')
        corpus = torch.load(fn)
    else:
        corpus = Corpus(datadir, dataset)


argmax = lambda x: np.argmax(x, 1)

pred_fns = {
    'text8': argmax,
}

filenames = {
    'text8': 'text8.tsv',
}

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    submit = 'store_true'
    dataset = 'text8'
    n_ctx = 512
    save_dir = 'save/'
    desc = 'Description'
    data_dir = 'data/'
    log_dir = 'log/'
    submission_dir = 'submission/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('device', device, 'n_gpu', n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)))
    text_encoder = TextEncoder("models/", "models/")  # !!!!!!!!
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    print("Encoding dataset...")
    ((trX1, trX2, trX3, trY),
     (vaX1, vaX2, vaX3, vaY),
     (teX1, teX2, teX3)) = encode(stories(data_dir, n_valid=374),
                                  encoder=text_encoder)
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_special = 3
    max_len = n_ctx // 2 - 2
    n_ctx = min(max(
        [len(x1[:max_len]) + max(len(x2[:max_len]),
                                 len(x3[:max_len])) for x1, x2, x3 in zip(trX1, trX2, trX3)]
        + [len(x1[:max_len]) + max(len(x2[:max_len]),
                                   len(x3[:max_len])) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]
        + [len(x1[:max_len]) + max(len(x2[:max_len]),
                                   len(x3[:max_len])) for x1, x2, x3 in zip(teX1, teX2, teX3)]
    ) + 3, n_ctx)
    vocab = n_vocab + n_special + n_ctx
    trX, trM = transform_roc(trX1, trX2, trX3)
    vaX, vaM = transform_roc(vaX1, vaX2, vaX3)
    if submit:
        teX, teM = transform_roc(teX1, teX2, teX3)

    n_train = len(trY)
    n_valid = len(vaY)
    n_batch_train = 8 * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * 3

    dh_model = Model(clf_token, 'multiple_choice', vocab, n_ctx)

    criterion = nn.CrossEntropyLoss(reduce=False)
    model_opt = torch.optim.adam(dh_model.parameters, lr=6.25e-5, )

    compute_loss_fct = MultipleChoiceLossCompute(criterion,
                                                 criterion,
                                                 0.5,
                                                 model_opt)

    dh_model.to(device)
    dh_model = nn.DataParallel(dh_model)

    n_updates = 0
    n_epochs = 0
    if dataset != 'stsb':
        trYt = trY
    if submit:
        path = os.path.join(save_dir, desc, 'best_params')
        torch.save(dh_model.state_dict(), make_path(path))
    best_score = 0
    for i in range(3):
        print("running epoch", i)
        run_epoch()
        n_epochs += 1
        log(save_dir, desc)

    path = os.path.join(save_dir, desc, 'best_params')
    dh_model.load_state_dict(torch.load(path))
    predict(dataset, submission_dir)
