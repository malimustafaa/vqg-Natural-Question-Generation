"""
beam_search.py
---------------------------------
Beam search decoding for the GRNN, width 8 (paper default), with <unk>
banned from the output at test time (paper: "we do not allow the decoder to
produce this token at test time"). Ranks on raw cumulative log-probability
by default -- the paper doesn't specify either way -- but every extra token
multiplies in another sub-1 probability, so raw log-prob structurally favors
short sequences regardless of content quality. Pass length_penalty > 0 (GNMT-
style, Wu et al. 2016) to counteract that; 0.0 (default) preserves the
original paper-matching behavior exactly.
"""
import torch
import torch.nn.functional as F


def _length_norm(length, alpha):
    return ((5 + length) / 6) ** alpha


@torch.no_grad()
def beam_search_decode(model, fc7_feature, vocab, beam_width=8, max_len=20, length_penalty=0.0):
    device = fc7_feature.device
    h0 = model.init_hidden(fc7_feature.unsqueeze(0))  # (1, 1, hidden)

    # beam: (cumulative_logprob, token_id_seq, hidden_state, finished)
    beams = [(0.0, [vocab.sos_id], h0, False)]

    def ranking_score(logp, seq):
        # Applied at every prune step, not just the final pick -- otherwise
        # short sequences already win the mid-search comparisons (a finished
        # 4-token beam vs. a still-growing 7-token one) before normalization
        # ever gets a say.
        return logp / _length_norm(len(seq), length_penalty)

    for _ in range(max_len):
        if all(finished for _, _, _, finished in beams):
            break
        candidates = []
        for logp, seq, hid, finished in beams:
            if finished:
                candidates.append((logp, seq, hid, True))
                continue
            last_token = torch.tensor([seq[-1]], device=device)
            logits, new_hidden = model.step(last_token, hid)
            logits = logits.squeeze(0).clone()
            logits[vocab.unk_id] = float("-inf")
            log_probs = F.log_softmax(logits, dim=-1)
            topk_logp, topk_idx = log_probs.topk(beam_width)
            for step_logp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                new_seq = seq + [idx]
                candidates.append((logp + step_logp, new_seq, new_hidden, idx == vocab.eos_id))

        candidates.sort(key=lambda c: ranking_score(c[0], c[1]), reverse=True)
        beams = candidates[:beam_width]

    beams.sort(key=lambda b: ranking_score(b[0], b[1]), reverse=True)
    best_score, best_seq = beams[0][0], beams[0][1]
    tokens = vocab.decode(best_seq[1:], stop_at_eos=True)  # drop leading <sos>
    return " ".join(tokens), best_score
