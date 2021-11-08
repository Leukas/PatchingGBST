import torch

def mask_span_sentinel(
    sequence: torch.LongTensor, 
    mask_p: float=0.15, 
    poisson_lambda: float=3.5, 
    sentinel_range: tuple=(259, 359)):
    """ 
    Span masking as done in T5.
    Params:
        sequence: the sequence to mask
        mask_p: % of tokens masked on average 
        poisson_lambda: average length of masked spans
        sentinel_range: index range (exclusive) of the sentinels
    Returns:
        final_sequence: the masked sequence
        target_sequence: the masked spans
    """
    num_masked = torch.ceil(torch.Tensor([sequence.size(0) * mask_p])).long()
    span_lengths = torch.poisson(torch.full((sequence.size(0),), poisson_lambda-1)).long()+1

    # check if new seq will be longer, if so reroll (to avoid mem issues in training)
    longer_seq = span_lengths[span_lengths==0].numel() > (span_lengths[span_lengths!=0]-1).sum()
    
    # check if lengths is not long enough, if so reroll
    not_enough_masked = span_lengths.sum() < num_masked

    while longer_seq or not_enough_masked:
        span_lengths = torch.poisson(torch.full(sequence.size(), poisson_lambda-1)).long()+1
        longer_seq = span_lengths[span_lengths==0].numel() > (span_lengths[span_lengths!=0]-1).sum()
        not_enough_masked = span_lengths.sum() < num_masked

    # trim last if too long    
    total_lengths = torch.cumsum(span_lengths, dim=0).long()
    cutoff = torch.searchsorted(total_lengths, num_masked)
    span_lengths[cutoff] -= total_lengths[cutoff] - num_masked
    span_lengths = span_lengths[:cutoff+1]

    # get random idxs for masking, mask first idx
    idxs = torch.randperm(sequence.size(0))[:len(span_lengths)].sort()[0]

    masked_sequence = sequence.clone()
    assert -1 not in masked_sequence
    masked_sequence[idxs] = -1

    target_sequence = []

    # mark additional tokens for removal
    keep = torch.ones_like(sequence).bool()
    last_end = 0
    filtered_idxs = []
    for i, span in enumerate(span_lengths):
        keep[idxs[i]+1:idxs[i]+span.item()] = 0

        # if 2 spans are adjacent, combine into 1
        if last_end == idxs[i]:
            keep[idxs[i]] = 0
            last_end = idxs[i] + span.item()        
            target_sequence += sequence[idxs[i]:idxs[i] + span.item()]
        # if last span overlaps with next
        elif last_end > idxs[i]:
            if idxs[i] + span.item() > last_end:
                target_sequence += sequence[last_end:idxs[i] + span.item()]
                last_end = idxs[i] + span.item()
        else: # no overlap, proceed as normal
            target_sequence += [sentinel_range[0]+len(filtered_idxs)]
            target_sequence += sequence[idxs[i]:idxs[i]+span.item()]
            filtered_idxs.append(idxs[i])
            last_end = idxs[i] + span.item()

    target_sequence += [sentinel_range[0]+len(filtered_idxs)]

    # replace masks with sentinels
    # masks = masked_sequence==-1
    masks = torch.LongTensor(filtered_idxs).to(sequence.device)
    max_sentinels = sentinel_range[1] - sentinel_range[0]
    assert masks.numel() < max_sentinels
    sentinels = torch.arange(masks.numel(), device=sequence.device) + sentinel_range[0]
    # print(masks.size(), sentinels.size(), masked_sequence.size())
    masked_sequence[masks] = sentinels

    # remove tokens
    masked_sequence = masked_sequence[keep]

    target_sequence = torch.LongTensor(target_sequence).to(sequence.device)

    return (masked_sequence, target_sequence)