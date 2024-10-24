import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast


class Config:
    checkpoint = "/home/wanyao/wcloong/Locating-and-Updating-Code-Knowledge-for-Code-LM/hf-model/codeparrot"
    dif_step = 20 # differantial step, we transform the continuous differantial computing into discrete steps
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def single_step_for_neuron_score(
        X: str,
        Y: str,
        position: tuple,    # (layer, dim, index)
        model: GPT2LMHeadModel,
        tok: GPT2TokenizerFast,
        config: Config
):
    '''
    We divide the computing into m steps. Each step holds different paras for target token.
    In this method, we will calculate the score of one step, where the para of the neuron is froze.
    
    @para: X, the pre-text prompt
    @para: Y, the target output
    @para: position, a tuple with 3 elements, (layer, dim, index), to decide the position of one neuron
    @para: model, huggingface model, GPT2LMHeadModel
    @para: tok, huggingface tokenizer, GPT2TokenizerFast 
    '''
    model.to(config.device)
    for para in model.parameters():
        para.requires_grad = True
    model.train()

    token_x = tok(X, return_tensors='pt').input_ids.to(config.device)
    token_y = tok(Y, return_tensors='pt').input_ids.to(config.device)
    len_y = token_y.size(-1)

    target_probs = []   # store the probability in each inference
    target_grad = []    # store the grad of target neuron in each inference
    for i in range(len_y):
        input_ids = torch.cat((token_x, token_y[:, :i]), dim=-1)
        logits = model(input_ids).logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        target_prob = probs[-1, token_y[0, -1]]
        target_probs.append(target_prob)

        model.zero_grad()

        target_prob.backward()

        for name, para in model.named_parameters():
            if name == f'transformer.h.{position[0]}.mlp.c_proj.weight':
                target_grad.append(para[position[1], position[2]])
                break

    part_score = torch.sum(
        torch.tensor(target_grad, device=config.device) / 
        torch.tensor(target_prob, device=config.device))
    return part_score

def score_computing_for_single_neuron(
        X: str,
        Y: str,
        positoin: tuple,
        model: GPT2LMHeadModel,
        tok: GPT2TokenizerFast,
        config: Config
):
    model_copy = GPT2LMHeadModel(model.config)
    model_copy.load_state_dict(model.state_dict())

    orig_weight = model_copy.transformer.h[position[0]].mlp.c_proj.weight[position[1], position[2]]

    score = 0.
    for i in range(config.dif_step):
        new_weight = float(i / config.dif_step) * orig_weight
        model_copy.transformer.h[position[0]].mlp.c_proj.weight[position[1], position[1]] = new_weight

        score += single_step_for_neuron_score(X, Y, position, model_copy, tok, config)

    return score / (config.dif_step + 1) * orig_weight





config = Config()

X = '''import torch.nn as nn
def ETFPriceLoss(output, label):
    loss = torch.abs(torch.add('''

Y = '''label,[MASK]))'''

position = (1, 0, 0)

model = AutoModelForCausalLM.from_pretrained(config.checkpoint)
tok = AutoTokenizer.from_pretrained(config.checkpoint)

# single_step_for_neuron_score(X, Y, position, model, tok, config)

score_computing_for_single_neuron(X, Y, position, model, tok, config)
