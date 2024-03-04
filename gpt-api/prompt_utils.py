LLAMA_SYS = "Be helpful, respectful and honest. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

LLAMA_SYS_SHORT = "Be helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."

MPT_SYS = "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."

ALPACA_SYS = "Below is an instruction that describes a task. Write a response that appropriately completes the request."


def get_prompt_template(template):
    template_dict = {"system": "", "user": "%s"}
    if template == "vanilla":
        return template_dict
    elif template == "llamasys":
        template_dict["system"] = LLAMA_SYS
    elif template == "alpaca":
        template_dict["system"] = ALPACA_SYS
        template_dict["user"] = "### Instruction:\n%s\n\n### Response:\n"
    elif template == "mpt":
        template_dict["system"] = MPT_SYS
    elif template == "llamasys_short":
        template_dict["system"] = LLAMA_SYS_SHORT
    else:
        raise NotImplementedError
    return template_dict