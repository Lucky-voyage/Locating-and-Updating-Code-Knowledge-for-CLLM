# Datast

## Dataset Structure

### 1. **Taxonomy**
    
`API_Pair`: 每一个 API_Pair 包含一个 Outdated_API 的信息和一个 Updated_API 的信息，以及两个 API 的出处，即 Library。对于一个 API 的信息，只需要 Version (API 版本)以及 API_Signature (API 函数签名)

`Code_Bag`: Code_Bag 是一个列表，其中每一个元素称为一个 Code_Pair。
    
`Code_Pair`: 一共包含两部分，Outdated_API_Code 以及 Updated_API_Code。Outdated/Updated_API_Code 均为一个代码样例，称为 Code_Snippet

### 2. **Requirements**
   
- **Code_Pair**

   每一个 Code_Pair 包含两个不同代码片段，对比二者（主要针对 python 代码，**以行为单位**），不涉及 API 调用的部分为 `neutral part`，而涉及 API 调用的部分称为 `pivot part`。值得注意的是，neutral part 和 pivot part 的概念只在同一个 Code_Pair 中才有意义。

   - **Appropriate Calling**：Code_Pair 中的两个 Code_Snippet 必须都调用了目标 API，同时确保调用 **正确** 且 **有意义**。
   - **neutral part**：我们必须确保 neutral part 保持一致。
   - **pivot part**: 对于大多数代码片段，我们希望 pivot part 中存在不同之处，这是为了体现 Updated_API 的更新功能。另外，我们仍然 **允许** 部分 Code_Pair 的 pivot part 完全相同；对于相当数量的 API 更新，仅仅是对 API 的功能补充或者代码补丁，API 调用方法和功能没有发生本质变化。我们仍然需要基于 Outdated_API 调用能力来让模型掌握 Updated_API。

- **Code_Bag**

    Code_Bag 是一组 Code_Pair，设置多个 Code_Pair 的目的是为了 **全面** 的展现 Updated_API 的信息（包括 API 语法、语义、功能等诸多内容）。我们相信，充分的 Code_Pair 作为样例可以完美地隐式展现 API 知识的全貌。

    因此，我们对 Code_Bag 中的代码样例提出以下要求：

    - **Highlight**：Code_Bag 中的样例围绕 Updated_API 的要求更高。
    - **参数全面性**：Code_Bag 样例中对 Updated_API 的调用必须涉及所有的参数使用，从而确保在结构上完全长我 Updated_API 的信息。
    - **返回值全面性**： 如果 Updated_API 存在返回值，则必须有确保在相当比例的 Code_Pair 中，API 返回值得到了使用。

### 3. **Category**

我们的数据集需要涉及多种 API 知识的操作，包括 `editing`（修改），`inserting`（插入）、`erasing`（删除）。不同种类的 API 操作，Code_Pair 的结构也有区别。因此我们需要一个标签 `type` 来标注一个样本的种类。

- **Editing**
  
  Editing 操作必须包含 Updated_API 和 Outdated_API，从而确保模型的编辑是从 Updated_API 到 Outdated_API 发生的。

- **Inserting**

  Inserting 操作为插入知识，则不需要模型对 Outdated_API 有知识存储，因此 Code_Pair 中只需要 Updated_API 相关内容。

- **Erasing**
  
  Erasing 操作是删除已有知识，不存在 Updated_API。Code_Pair 中只需要包含 Outdated_API 相关内容。

### 4. **Paradigm**

Here is a paradigm for API `torch.autograd.grad` in version 2.0 and 2.5 without specific code snippets.
```json
{
    "type": "editing",
    "API_Pair":{
        "Libaray": "torch",
        "Outdated_API": {
            "Version": "torch 2.0",
            "API_Signature": "torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, is_grads_batched=False)->Tuple[Tensor, ...]"
        },
        "Updated_API": {
            "Version": "torch 2.5",
            "API_Signature": "torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=None, is_grads_batched=False, materialize_grads=False)->Tuple[Tensor, ...]"
        }
    },
    "Code_Pairs": [
        {
            "Outdated_API_Code": "Outdated_API_Code A",
            "Updated_API_Code": "Updated_API_Code A"
        },
        {
            "Outdated_API_Code": "Outdated_API_Code B",
            "Updated_API_Code": "Updated_API_Code B"
        }
    ]
}
```
