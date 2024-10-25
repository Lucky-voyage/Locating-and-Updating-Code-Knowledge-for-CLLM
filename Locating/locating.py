import copy
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast


class Config:
    checkpoint = "/home/wanyao/wcloong/Locating-and-Updating-Code-Knowledge-for-Code-LM/hf-model/codeparrot"
    dif_step = 20 # differantial step, we transform the continuous differantial computing into discrete steps
    top_k = 20 # sample the top-k neurons as pivot neurons
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def single_step_for_neuron_score(
        inputs: torch.tensor,
        position: tuple,    # (layer, dim, index)
        model: GPT2LMHeadModel,
        tok: GPT2TokenizerFast,
        config: Config
):
    '''
    We divide the computing into m steps. Each step holds different paras for target token.
    In this method, we will calculate the score of one step, where the para of the neuron is froze.
    
    Args:
        X(str): the pre-text prompt
        Y(str): the target output
        position(tuple): a tuple with 3 elements, (layer, dim, index), to decide the position of one neuron
        model(GPT2LMHeadModel)
        tok(GPT2TokenizerFast)
        config(Config)

    Returns:
        torch.tensor: the gradient of neuron with current value
    '''
    target_probs = []   # store the probability in each inference
    target_grad = []    # store the grad of target neuron in each inference
    
    logits = model(inputs).logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)

    target_prob = probs[:, token_y[0, -1]]
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
        position: tuple,
        model: GPT2LMHeadModel,
        tok: GPT2TokenizerFast,
        config: Config
):
    '''
    For each neuron, locating algorithm will conduct multiple steps to computing neuron's gradient with different value

    Args:
        X(str): the pre-text prompt
        Y(str): the target output
        position(tuple): a tuple with 3 elements, (layer, dim, index), to decide the position of one neuron
        model(GPT2LMHeadModel)
        tok(GPT2TokenizerFast)
        config(Config)

    Returns:
        torch.tensor: the score of one neuron  
    
    '''
    model.to(config.device)
    for para in model.parameters():
        para.requires_grad = True
    model.train()

    model_copy = copy.deepcopy(model)
    orig_weight = model.transformer.h[position[0]].mlp.c_proj.weight[position[1], position[2]].clone()
    
    token_x = tok(X, return_tensors='pt').input_ids.to(config.device)
    token_y = tok(Y, return_tensors='pt').input_ids.to(config.device)
    len_y = token_y.size(-1)
    input_batch = []
    target_token = []
    for i in range(len_y - 1):
        input_ids = torch.cat((token_x, token_y[:, :i]), dim=-1)
        input_batch.append(input_ids)
        target_token.append(token_y[:, i])
    
    score = 0.
    import time
    a = time.time()
    for i in range(config.dif_step):
        new_weight = float(i / config.dif_step) * orig_weight
        with torch.no_grad():
            model_copy.transformer.h[position[0]].mlp.c_proj.weight[position[1], position[1]] = new_weight

        score += single_step_for_neuron_score(X, Y, position, model_copy, tok, config)
    b = time.time()
    print(f'time: {b - a}')
    return score / (config.dif_step + 1) * orig_weight


def locating_pivot_neurons(
    X: str,
    Y: str,
    model: GPT2LMHeadModel,
    tok: GPT2TokenizerFast,
    config: Config
):

    return




config = Config()

X = '''import torch.nn as nn
def ETFPriceLoss(output, label):
    loss = torch.abs(torch.add('''

Y = '''label,[MASK]))'''

position = (1, 0, 0)

model = AutoModelForCausalLM.from_pretrained(config.checkpoint)
tok = AutoTokenizer.from_pretrained(config.checkpoint)

# single_step_for_neuron_score(X, Y, position, model, tok, config)

print(score_computing_for_single_neuron(X, Y, position, model, tok, config))
