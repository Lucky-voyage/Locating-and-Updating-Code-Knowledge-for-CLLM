# Locating
## Input Template


We divide each code snippet into 3 parts, `pre_text`, `post_text` and `mask`.

**Paradigm**
```python
# Pre text
import torch.nn as nn

def ETFPriceLoss(output, label):
    loss = torch.abs(torch.add(label,-1, output))
    return loss

# Post text
import torch.nn as nn

def ETFPriceLoss(output, label):
    loss = torch.abs(torch.add(label,output, alpha=-1))
    return loss

# Mask text
# We have 3 masking levels
import torch.nn as nn

def ETFPriceLoss(output, label):
    loss = torch.abs(torch.add(label,[MASK]))
    return loss
```
We can try different mask strategies or combine multiple methods

## Score Definition

1. Define the probability of **generating specific tokens with given prompt**
   
    We use $X$ to denote Pre_text sequence, and $Y$ to denote target generating sequence. $|Y|$ is the length of $Y$, and $y_i (i \in [1, {Y}])$ represents the $i_{th}$ token in sequence $Y$.

    We apply the **chain rule** to define the probability:

    $$P(Y|X)=P(y_1|X)\times P(y_2|X+y_1)\times ... P(y_{|Y|}|X+\sum_{i=1}^{|Y|-1}y_i)$$

    where $P(y_{|Y|}|X+\sum_{i=1}^{k}y_i)$ is the conditional probabilty while the prompt is the sequence concatenating $X$ and $(y_1, y_2, ... , y_k)$. And we can compute the conditional probability via logits.

    To address the problem of numerical underflow, we apply log probability. We have:

    $$\log P(Y|X) = \sum_{i=1}^{|Y|} \log P(y_{i}|X+Y[0:i-1])$$

2. Apply **gradient attribution** or **gradient accumulation**

    We apply gradient accumulation method to evaluate the contribution of one nueron to the generated sequence. We set the baseline of accumulation accumulation to $0$.Therefore, we have the formula:

    $$ IG_{w}=w\int_{\alpha=0}^1 \frac{\partial F(\alpha \cdot w)}{\partial w} dw$$

    We transform the formula to the discrete form, with $m$ steps.

    $$ IG_{w} = \frac{w}{m+1} \sum_{i=0}^{m} \frac{\partial F(\frac{i\cdot w}{m})}{\partial w}$$

    Substitute the formula:

    $$IG_{w} = \frac{w}{m+1} \sum_{i=0}^{m} \frac{\partial \sum_{j=1}^{|Y|}\log P(y_j|X+Y[0:j-1])}{\partial w}\\=\frac{w}{m+1} \sum_{i=0}^{m} \sum_{j=1}^{|Y|}\frac{\partial \log P(y_j|X+Y[0:j-1])}{\partial w}\\= \frac{w}{m+1} \sum_{i=0}^{m} \sum_{j=1}^{|Y|}\frac{\partial P(y_j|X+Y[0:j-1])}{P(y_j|X+Y[0:j-1]) \partial w}$$

    We can compute $P(y_j|X+Y[0:i-1])$ via logits. Therefore,

    $$IG_{w}= \frac{w}{m+1} \sum_{i=0}^{m} \sum_{j=1}^{|Y|}\frac{model(X+Y[0:j-1]).grad(w)}{logits(X+Y[0:j-1])}$$
