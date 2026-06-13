
# Adding a New LLM to Agent

This guide explains how to add support for a new LLM and use it within the Agent framework. In this example, we add the **OpenChat-3.5** model from HuggingFace.

> **Note:** Make sure you have installed all dependencies of Agent by following the installation guide!

---

## 🛠️ Create a New LLM Config

The simplest way to add a new model is via HuggingFace, either by loading it from the hub or from a local path. To do this, add a new config under:

```
./configs/llm/hf/
```

Update the config with the model ID and context length. You can also change `model_id` to a locally downloaded model path to avoid download times.

Here’s an example configuration for **OpenChat-3.5**:

```python
llm_config = """ 
defaults:
  - hf

model_id: openchat/openchat-3.5-0106
context_length: 8192
tokenizer_kwargs: {}
"""

with open("./configs/llm/hf/example_openchat-3.5.yaml", "w") as file:
    file.write(llm_config)
```

Now your Agent setup can support and run OpenChat-3.5 or any other HuggingFace-compatible LLM!

To know more about other type of backends supported by Agent 
K and hosting llms follow this [guide](https://github.com/huawei-noah/HEBO/blob/master/Agent/docs/source/llms.rst).