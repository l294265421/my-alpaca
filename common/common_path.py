import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# project_dir = '/my-alpaca/MyDrive/my-alpaca'

alpaca_lora_dir = os.path.join(project_dir, 'alpaca_lora')

if __name__ == '__main__':
    print(project_dir)
