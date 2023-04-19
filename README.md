# my-alpaca
Try original alpaca. The multi-turn version is at [multi-turn-alpaca](https://github.com/l294265421/multi-turn-alpaca) and the version further trained with RLHF (Reinforcement Learning with Human Feedback) is at [alpaca-rlhf](https://github.com/l294265421/alpaca-rlhf). The trained multi-turn alpaca is amazing. There is a chat case with the multi-turn alpaca:

![](./figures/chat.jpg)

## Step by Step
### Colab (Free colab is too slow to perform one epoch)
- [finetune](my_alpaca/colab/finetune.ipynb)

### [Autodl](https://www.autodl.com/home)
- [filetune](my_alpaca/autodl/finetune.py)
  - nohup sh run.sh my_alpaca/autodl/finetune.py > autodl.log 2>&1 &
- [inference_llama](my_alpaca/autodl/inference_llama.py)
  - sh run.sh my_alpaca/autodl/inference_llama.py
  ![](./figures/autodl_llama.png)
- [inference_llama_gradio](my_alpaca/autodl/inference_llama_gradio.py)
  - sh run.sh my_alpaca/autodl/inference_llama_gradio.py
- [inference_alpaca_lora](my_alpaca/autodl/inference_alpaca_lora.py)
  - sh run.sh my_alpaca/autodl/inference_alpaca_lora.py
  ![](./figures/autodl_llama_lora.png)
- [inference_alpaca_lora_gradio](my_alpaca/autodl/inference_alpaca_lora_gradio.py)
    - sh run.sh my_alpaca/autodl/inference_alpaca_lora_gradio.py
    
- Tips
  - Autodl的网速特别慢，没法下llama，只能慢慢从本地传。制作了镜像，包含了llama模型，有需要的话，可以留下autodl账号，我在平台上分享（貌似只能指定账号分享）。

## References

### Pre-trained Language Models
- LLaMA
  - [Paper](./papers/2023-LLaMA-%20Open%20and%20Efficient%20Foundation%20Language%20Models.pdf)
  - [Model](https://huggingface.co/decapoda-research)
- GLM
  - [GitHub](https://github.com/THUDM/GLM)
  - [Paper](./papers/2023-ICLR-GLM-130B-%20An%20Open%20Bilingual%20Pre-trained%20Model.pdf)

### Datasets
#### For Pre-training
- 大规模中文自然语言处理语料 Large Scale Chinese Corpus for NLP
  - [GitHub](https://github.com/brightmart/nlp_chinese_corpus)
- CSL: A Large-scale Chinese Scientific Literature Dataset 中文科学文献数据集
  - [GitHub](https://github.com/ydli-ai/CSL)
- 中文图书语料集合
  - [GitHub](https://github.com/FudanNLPLAB/CBook-150K)

#### For Fine-tuning
- ChatAlpaca
  - [GitHub](https://github.com/cascip/ChatAlpaca)
- GPT-4-LLM
  - [GitHub](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
  - [Paper](https://arxiv.org/pdf/2304.03277.pdf)
  - [Site](https://instruction-tuning-with-gpt-4.github.io/)

### Fine-tuning

#### General
- Alpaca: A Strong, Replicable Instruction-Following Model
  - [Site](https://crfm.stanford.edu/2023/03/13/alpaca.html)
  - [GitHub](https://github.com/tatsu-lab/stanford_alpaca#fine-tuning)
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality
  - [GitHub](https://github.com/lm-sys/FastChat)
  - [Site](https://vicuna.lmsys.org/)
  - [Online Demo](https://chat.lmsys.org/)
- Koala: A Dialogue Model for Academic Research
  - [Blog](https://bair.berkeley.edu/blog/2023/04/03/koala/)
  - GitHub
      - [Koala_data_pipeline](https://github.com/young-geng/koala_data_pipeline)
      - [Koala Evaluation Set](https://github.com/arnav-gudibande/koala-test-set)
- alpaca-lora
  - [GitHub](https://github.com/tloen/alpaca-lora)
- ChatGLM-6B
  - [GitHub](https://github.com/THUDM/ChatGLM-6B)
  - [Blog](https://chatglm.cn/blog)
- Firefly
  - [GitHub](https://github.com/yangjianxin1/Firefly)
- thai-buffala-lora-7b-v0-1
  - [Model](https://huggingface.co/Thaweewat/thai-buffala-lora-7b-v0-1)
- multi-turn-alpaca
  - [GitHub](https://github.com/l294265421/multi-turn-alpaca)
- Open-Assistant
  - [Site](https://open-assistant.io/zh)
  - [GitHub](https://github.com/LAION-AI/Open-Assistant)
  - [Paper](./papers/2023-OpenAssistant%20Conversations%20-%20Democratizing%20Large%20Language%20Model%20Alignment.pdf)

#### Chinese
- Chinese-ChatLLaMA
  - [GitHub](https://github.com/ydli-ai/Chinese-ChatLLaMA)
  - Blog
    - [训练中文LLaMA大规模语言模型](https://zhuanlan.zhihu.com/p/612752963)
    - [ChatLLaMA：用指令微调训练中文对话大模型](https://zhuanlan.zhihu.com/p/616748134)
- BELLE
  - [GitHub](https://github.com/LianjiaTech/BELLE)
- Chinese-LLaMA-Alpaca
  - [GitHub](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- Luotuo-Chinese-LLM
  - [GitHub](https://github.com/LC1332/Luotuo-Chinese-LLM)
- Chinese-Vicuna
  - [GitHub](https://github.com/Facico/Chinese-Vicuna)
- Chinese-alpaca-lora
  - [GitHub](https://github.com/LC1332/Chinese-alpaca-lora)

#### Japanese
- Japanese-Alpaca-LoRA
  - [GitHub](https://github.com/kunishou/Japanese-Alpaca-LoRA)

#### Medical
- 2023-ChatDoctor: A medical chat model fine-tuned on llama model using medical domain knowledge
  - [Paper](./papers/2023-ChatDoctor-%20A%20Medical%20Chat%20Model%20Fine-tuned%20on%20LLaMA%20Model%20using%20Medical%20Domain%20Knowledge.pdf)
- 华驼(HuaTuo): 基于中文医学知识的LLaMA微调模型
  - [GitHub](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)

#### Law
- LawGPT_zh：中文法律大模型（獬豸）
  - [GitHub](https://github.com/LiuHC0428/LAW-GPT)

#### Recommendation
- 2023-Recalpaca: Low-rank llama instruct-tuning for recommendation

### Evaluation
- LLMZoo: a project that provides data, models, and evaluation benchmark for large language models.
  - [GitHub](https://github.com/FreedomIntelligence/LLMZoo)
  
### Tools
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
  - [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
    - [debug-deepspeed-chat](https://github.com/l294265421/debug-deepspeed-chat)
- [ColossalAI](https://github.com/hpcaitech/ColossalAI)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

### Interfaces
- [ChuanhuChatGPT](https://github.com/GaiZhenbiao/ChuanhuChatGPT/)
- [ChatGLM-webui](https://github.com/Akegarasu/ChatGLM-webui)

### Products
- [ChatGPT](https://chat.openai.com/)
- [文心一言](https://yiyan.baidu.com/)
- [通义千问](https://tongyi.aliyun.com/)
- AgentGPT
  - [GitHub](https://github.com/reworkd/AgentGPT)
- HuggingGPT
  - [GitHub](https://github.com/microsoft/JARVIS)
  - [Paper](https://arxiv.org/abs/2303.17580)
- AutoGPT
  - [GitHub](https://github.com/Significant-Gravitas/Auto-GPT)
- MiniGPT-4
  - [GitHub](https://github.com/Vision-CAIR/MiniGPT-4)
- ShareGPT
  - [GitHub](https://github.com/domeccleston/sharegpt)

### Preliminary
#### Papers
- 2020-Integer Quantization for Deep Learning Inference Principles and Empirical Evaluation [[paper](./papers/2020-Integer%20Quantization%20for%20Deep%20Learning%20Inference%20Principles%20and%20Empirical%20Evaluation.pdf)]
- 2021-LoRA- Low-Rank Adaptation of Large Language Models [[paper](./papers/2021-LoRA-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models.pdf)]
- 2022-SELF-INSTRUCT- Aligning Language Model with Self Generated Instructions [[paper](./papers/2022-SELF-INSTRUCT-%20Aligning%20Language%20Model%20with%20Self%20Generated%20Instructions.pdf)]
- 2023-LLaMA- Open and Efficient Foundation Language Models [[paper](./papers/2023-LLaMA-%20Open%20and%20Efficient%20Foundation%20Language%20Models.pdf)]
- 2023-Llama-adapter: Efficient fine-tuning of language models with zero-init attention
- 2023-RRHF: Rank Responses to Align Language Models with Human Feedback without tears [[paper](./papers/2023-RRHF-%20Rank%20Responses%20to%20Align%20Language%20Models%20with%20Human%20Feedback%20without%20tears.pdf)] [[code](https://github.com/GanjinZero/RRHF)]

##### Neural Text Generation (For Better Generation)
- 2020-ICLR-Neural text generation with unlikelihood training [[paper](./papers/2020-ICLR-Neural%20text%20generation%20with%20unlikelihood%20training.pdf)]
- 2021-ICLR-Mirostat- a neural text decoding algorithm that directly controls perplexity [[paper](./papers/2021-ICLR-Mirostat-%20a%20neural%20text%20decoding%20algorithm%20that%20directly%20controls%20perplexity.pdf)]
- 2022-NIPS-A Contrastive Framework for Neural Text Generation [[paper](./papers/2022-NIPS-A%20Contrastive%20Framework%20for%20Neural%20Text%20Generation.pdf)]

#### Articles
- [ChatGPT-Techniques-Introduction-for-Everyone](https://github.com/l294265421/ChatGPT-Techniques-Introduction-for-Everyone)
- 分布式训练
  - [Pytorch 分布式训练](https://zhuanlan.zhihu.com/p/76638962)
  - [浅谈Tensorflow分布式架构：ring all-reduce算法](https://zhuanlan.zhihu.com/p/69797852)
  - [Optimizer state sharding (ZeRO)](https://zhuanlan.zhihu.com/p/394064174)
    - [ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)

### Other Resources