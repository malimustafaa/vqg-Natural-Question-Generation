"""
beam_search.py
---------------------------------
Beam search decoding for the GRNN, width 8 (paper default), with <unk>
banned from the output at test time (paper: "we do not allow the decoder to
produce this token at test time"). No length normalization -- the paper
doesn't mention any, so beams are ranked on raw cumulative log-probability.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def beam_search_decode(model, fc7_feature, vocab, beam_width=8, max_len=20):
    device = fc7_feature.device
    h0 = model.init_hidden(fc7_feature.unsqueeze(0))  # (1, 1, hidden)

    # beam: (cumulative_logprob, token_id_seq, hidden_state, finished)
    beams = [(0.0, [vocab.sos_id], h0, False)]

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

        candidates.sort(key=lambda c: c[0], reverse=True)
        beams = candidates[:beam_width]

    beams.sort(key=lambda b: b[0], reverse=True)
    best_score, best_seq = beams[0][0], beams[0][1]
    tokens = vocab.decode(best_seq[1:], stop_at_eos=True)  # drop leading <sos>
    return " ".join(tokens), best_score
