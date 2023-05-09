# my-alpaca
Reproduce alpaca

## Relate Repositories
- [my-llm](https://github.com/l294265421/my-llm) all about large language models
- [try-large-models](https://github.com/l294265421/try-large-models) try large models
- [multi-turn-alpaca](https://github.com/l294265421/multi-turn-alpaca) train alpaca with multi-turn dialogue datasets
- [alpaca-rlhf](https://github.com/l294265421/alpaca-rlhf) train multi-turn alpaca with RLHF (Reinforcement Learning with Human Feedback)

## Step by Step
- [filetune](my_alpaca/autodl/finetune.py)
  - nohup sh run.sh my_alpaca/autodl/finetune.py > autodl.log 2>&1 &
- [inference_llama_gradio](my_alpaca/autodl/inference_llama_gradio.py)
  - sh run.sh my_alpaca/autodl/inference_llama_gradio.py
- [inference_alpaca_lora_gradio](my_alpaca/autodl/inference_alpaca_lora_gradio.py)
    - sh run.sh my_alpaca/autodl/inference_alpaca_lora_gradio.py
