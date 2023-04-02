from transformers import LlamaTokenizer


def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


if __name__ == '__main__':
    base_model: str = "decapoda-research/llama-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    vocab = tokenizer.get_vocab()
    chinese_chars = {}
    for word, index in vocab.items():
        if is_contains_chinese(word):
            chinese_chars[word] = index
    # 700个汉字
    print(chinese_chars)
