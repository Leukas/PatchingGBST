import torch
import numpy as np
import random

def shuffle_segments(segs, unmasked_tokens):
    """
    We control 20% mask segment is at the start of sentences
                20% mask segment is at the end   of sentences
                60% mask segment is at random positions,
    """

    p = np.random.random()
    if p >= 0.8:
        shuf_segs = segs[1:] + unmasked_tokens
    elif p >= 0.6:
        shuf_segs = segs[:-1] + unmasked_tokens
    else:
        shuf_segs = segs + unmasked_tokens

    random.shuffle(shuf_segs)
    
    if p >= 0.8:
        shuf_segs = segs[0:1] + shuf_segs
    elif p >= 0.6:
        shuf_segs = shuf_segs + segs[-1:]
    return shuf_segs

def get_segments(mask_len, span_len):
    segs = []
    while mask_len >= span_len:
        segs.append(span_len)
        mask_len -= span_len
    if mask_len != 0:
        segs.append(mask_len)
    return segs


def unfold_segments(segs):
    """Unfold the random mask segments, for example:
        The shuffle segment is [2, 0, 0, 2, 0], 
        so the masked segment is like:
        [1, 1, 0, 0, 1, 1, 0]
        [1, 2, 3, 4, 5, 6, 7] (positions)
        (1 means this token will be masked, otherwise not)
        We return the position of the masked tokens like:
        [1, 2, 5, 6]
    """
    pos = []
    curr = 1   # We do not mask the start token
    for l in segs:
        if l >= 1:
            pos.extend([curr + i for i in range(l)])
            curr += l
        else:
            curr += 1
    return np.array(pos)

def mask_word(w):
    _w_real = w
    _w_rand = np.random.randint(10000, size=w.shape)
    _w_mask = np.full(w.shape, 1)

    probs = torch.multinomial(torch.Tensor([0.8, 0.1, 0.1]), len(_w_real), replacement=True)

    _w = _w_mask * (probs == 0).numpy() + _w_real * (probs == 1).numpy() + _w_rand * (probs == 2).numpy()
    return _w

def restricted_mask_sent(x, l, span_len=10000, word_mass=0.5):
    """ Restricted mask sents
        if span_len is equal to 1, it can be viewed as
        discrete mask;
        if span_len -> inf, it can be viewed as 
        pure sentence mask
    """
    if span_len <= 0:
        span_len = 1
    max_len = 0
    positions, inputs, targets, outputs, = [], [], [], []
    mask_len = round(len(x[:, 0]) * word_mass)
    print("LEN:", mask_len)
    len2 = [mask_len for i in range(l.size(0))]
    
    unmasked_tokens = [0 for i in range(l[0] - mask_len - 1)]
    segs = get_segments(mask_len, span_len)

    print(segs)
    
    for i in range(l.size(0)):
        words = np.array(x[:l[i], i].tolist())
        shuf_segs = shuffle_segments(segs, unmasked_tokens)
        pos_i = unfold_segments(shuf_segs)
        output_i = words[pos_i].copy()
        target_i = words[pos_i - 1].copy()
        words[pos_i] = mask_word(words[pos_i])

        inputs.append(words)
        targets.append(target_i)
        outputs.append(output_i)
        positions.append(pos_i - 1)

    x1  = torch.LongTensor(max(l) , l.size(0)).fill_(0)
    x2  = torch.LongTensor(mask_len, l.size(0)).fill_(0)
    y   = torch.LongTensor(mask_len, l.size(0)).fill_(0)
    pos = torch.LongTensor(mask_len, l.size(0)).fill_(0)
    l1  = l.clone()
    l2  = torch.LongTensor(len2)
    for i in range(l.size(0)):
        x1[:l1[i], i].copy_(torch.LongTensor(inputs[i]))
        x2[:l2[i], i].copy_(torch.LongTensor(targets[i]))
        y[:l2[i], i].copy_(torch.LongTensor(outputs[i]))
        pos[:l2[i], i].copy_(torch.LongTensor(positions[i]))

    pred_mask = y != 0
    y = y.masked_select(pred_mask)

    # print(x1[:, 0], x2[:, 0])

    return x1, l1, x2, l2, y, pred_mask, pos


def mass_collate():
    pass


if __name__ == "__main__":
    a = torch.arange(1000).unsqueeze(1) + 100
    l = torch.full((1,), a.size(0))

    out = restricted_mask_sent(a, l)

    print(out[0].transpose(0, 1), out[4])