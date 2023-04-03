# my-alpaca
try alpaca

## Step By Step
### Colab (Free colab is too slow to perform one epoch)
- [finetune](my_alpaca/colab/finetune.ipynb)

### [Autodl](https://www.autodl.com/home)
- [filetune](my_alpaca/autodl/finetune.py)
  - nohup sh run.sh my_alpaca/autodl/finetune.py > autodl.log 2>&1 &
- [inference_llama](my_alpaca/autodl/inference_llama.py)
  - sh run.sh my_alpaca/autodl/inference_llama.py
  ![](./figures/autodl_llama.png)
- [inference_alpaca_lora](my_alpaca/autodl/inference_alpaca_lora.py)
  - sh run.sh my_alpaca/autodl/inference_alpaca_lora.py
  ![](./figures/autodl_llama_lora.png)
- [inference_gradio](my_alpaca/autodl/inference_gradio.py)
    - sh run.sh my_alpaca/autodl/inference_gradio.py 
        - https://470d2bd22d463d61d5.gradio.live
    
- Tips
  - Autodl的网速特别慢，没法下llama，只能慢慢从本地传。制作了镜像，包含了llama模型，有需要的话，可以留下autodl账号，我在平台上分享（貌似只能指定账号分享）。

## References

### Articles
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
    - [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca#fine-tuning)
- [GPT fine-tune实战： 训练我自己的 ChatGPT](https://zhuanlan.zhihu.com/p/616504594?utm_source=wechat_session&utm_medium=social&utm_oi=556103293550534656)
- [Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality](https://vicuna.lmsys.org/)
    - [FastChat](https://github.com/lm-sys/FastChat)
    - [Chat with Open Large Language Models](https://chat.lmsys.org/)

### Repositories
- [llama](https://github.com/facebookresearch/llama)
- [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca#fine-tuning)
- [alpaca-lora](https://github.com/tloen/alpaca-lora)
- [FastChat](https://github.com/lm-sys/FastChat)
- [Chinese-ChatLLaMA](https://github.com/ydli-ai/Chinese-ChatLLaMA)

### Papers
- 2021-LoRA- Low-Rank Adaptation of Large Language Models [[paper](./papers/2021-LoRA-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models.pdf)]
- 2022-SELF-INSTRUCT- Aligning Language Model with Self Generated Instructions [[paper](./papers/2022-SELF-INSTRUCT-%20Aligning%20Language%20Model%20with%20Self%20Generated%20Instructions.pdf)]
- 2023-LLaMA- Open and Efficient Foundation Language Models [[paper](./papers/2023-LLaMA-%20Open%20and%20Efficient%20Foundation%20Language%20Models.pdf)]
