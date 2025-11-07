---
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
library_name: peft
model_name: smollm2-mbpp-rloo
tags:
- base_model:adapter:HuggingFaceTB/SmolLM2-360M-Instruct
- lora
- rloo
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for smollm2-mbpp-rloo

This model is a fine-tuned version of [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with RLOO, a method introduced in [Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs](https://huggingface.co/papers/2402.14740).

### Framework versions

- PEFT 0.17.1
- TRL: 0.24.0
- Transformers: 4.57.1
- Pytorch: 2.5.1+cu121
- Datasets: 4.2.0
- Tokenizers: 0.22.1

## Citations

Cite RLOO as:

```bibtex
@inproceedings{ahmadian2024back,
    title        = {{Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs}},
    author       = {Arash Ahmadian and Chris Cremer and Matthias Gall{'{e}} and Marzieh Fadaee and Julia Kreutzer and Olivier Pietquin and Ahmet {"{U}}st{"{u}}n and Sara Hooker},
    year         = 2024,
    booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), {ACL} 2024, Bangkok, Thailand, August 11-16, 2024},
    pages        = {12248--12267},
    publisher    = {Association for Computational Linguistics},
    editor       = {Lun{-}Wei Ku and Andre Martins and Vivek Srikumar},
}
```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```