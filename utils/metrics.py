# metrics.py
import sys
import numpy as np
from datasets import load_metric

bleu = load_metric('sacrebleu')
def compute_bleu(eval_pred, args):
    predictions, labels = eval_pred

    char_preds = []
    char_labels = []
    for b in range(predictions.shape[0]):
        pred = predictions[b]
        if (pred==args.eos_token_id).sum() > 0:
            eos_idx = np.argmax(pred==args.eos_token_id)
            pred = pred[:eos_idx]

        lab = labels[b]
        if (pred==args.eos_token_id).sum() > 0:
            eos_idx = np.argmax(lab==args.eos_token_id)
            lab = lab[:eos_idx]
        else:
            lab = lab[lab!=-100]

        if b < 5:
            print("P:", pred)
            print("L:", lab)
        char_preds.append(args.tok.decode(pred, skip_special_tokens=True))
        char_labels.append([args.tok.decode(lab, skip_special_tokens=True)])

    print("\npred:", char_preds[0:5])
    print("\nlabel:", char_labels[0:5])

    bleu_score = {"bleu": bleu.compute(predictions=char_preds, references=char_labels)['score']}

    sys.stdout.flush()
    return bleu_score