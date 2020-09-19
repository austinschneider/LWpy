import numpy as np
import functools

def chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def pairwise_merge(blocks_0, blocks_1, equal, merge):
    b_final = []
    b0_final = []
    b1_final = blocks_1
    for b0 in blocks_0:
        b1_left = []
        done = False
        for b1 in b1_final:
            if not done and equal(b0, b1):
                b_final.append(merge(b0, b1))
                done = True
            else:
                b1_left.append(b1)
        b1_final = b1_left
        if not done:
            b0_final.append(b0)
    return b_final + b0_final + b1_final

def block_merge(block_0, block_1):
    s0, d0 = block_0
    s1, d1 = block_1
    d = dict(d0)
    if "events" in d:
        d.update({"events": d0["events"]+d1["events"]})
    return s0, d

def block_equal(block_0, block_1):
    s0, v0, d0 = block_0
    s1, v1, d1 = block_1
    if v0 != v1:
        return False
    if s0 != s1:
        return False
    if s0 == "VolumeInjectionConfiguration" or s0 == "RangedInjectionConfiguration":
        pass
    elif s0 == "EnumDef":
        name0, d0 = d0
        name1, d1 = d1
        if name0 != name1:
            return False
    k_d0 = list(d0.keys())
    k_d1 = list(d1.keys())
    if len(k_d0) != len(k_d1):
        return False
    keys = np.unique(k_d0 + k_d1)
    return functools.reduce(lambda a,b: a and b, [d0[k]==d1[k] for k in keys])

def merge_blocks(blocks):
    block_lists = [[b] for b in blocks]
    while len(block_lists) > 1:
        next_block_lists = []
        for c in chunk(block_lists, 2):
            if len(c) == 1:
                next_block_lists.append(c[0])
            elif len(c) == 2:
                next_block_lists.append(pairwise_merge(c[0], c[1], block_equal, block_merge))
        block_lists = next_block_lists
    return block_lists[0]
