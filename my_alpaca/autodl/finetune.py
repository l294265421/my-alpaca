import fire

from alpaca_lora import finetune


if __name__ == '__main__':
    fire.Fire(finetune.train)
