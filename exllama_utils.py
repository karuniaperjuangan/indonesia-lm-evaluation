import torch
import torch.nn.functional as F
import numpy as np

import numpy as np

def softmax(x):
    z = x - np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    softmax = numerator/denominator
    return softmax

def predict_classification_causal_single_token(model, tokenizer, input_text, options,loras=None):
    #choices = ['A', 'B', 'C', 'D', 'E'][:len(labels)]
    #print(choices)
    option_ids = [tokenizer.encode(option)[-1] for option in options]
    with torch.no_grad():
        inputs = tokenizer.encode(input_text,add_bos=True)
        outputs = model.forward(inputs,loras=loras)
        last_token_logits = outputs[:, -1, :]
        choice_logits = last_token_logits[:, option_ids].detach().cpu().numpy()
        conf = softmax(choice_logits)
        pred = []
        for i in range(len(conf)):
            pred.append(dict(enumerate(options))[np.argmax(choice_logits[i])]) # decode the answer
    return conf, pred

def predict_classification_multi_token(model, tokenizer, input_text, options,loras=None):
    ppl_list = []
    for option in options:
        input_ids = tokenizer.encode(input_text+option,add_bos=True)
        target_ids = input_ids[:,1:]
        #logits adalah hasil dari model.forward(input_ids) yang merupakan logit dari prediksi token selanjutnya
        logits = model.forward(input_ids,loras=loras)
        logprobs = (torch.log_softmax(logits, dim=-1))[:,:-1,:]
        # Fungsi di bawah berfungsi untuk mengambil log probabilitas dari token yang seharusnya dihasilkan
        logprobs = torch.gather(logprobs, 2, target_ids.to(logprobs.device).unsqueeze(-1)).squeeze(-1)

        #convert to perplexity
        cel = -logprobs.mean(dim=-1)
        ppl = torch.exp(cel).detach().cpu().numpy()
        ppl_list.append(*ppl)
    ppl_list = np.array(ppl_list)
    choice_id = np.argmin(ppl_list)
    return ppl_list, options[choice_id]