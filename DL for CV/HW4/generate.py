import torch.nn.functional as F
import utils
import torch


def generate(model, init_seq, predict_len, top_k=5):
    current_seq = init_seq
    predicted = []
    for _ in range(predict_len):
        output = model(current_seq)

        # get the next word
        p = F.softmax(output, dim=1).data
        p, top_i = p.topk(top_k)
        word_i = top_i[0, torch.multinomial(p, 1).item()]
        predicted.append(word_i.item())

        # Update sequence
        current_seq = torch.roll(current_seq, -1, 1)
        current_seq[-1, -1] = word_i

    return predicted







