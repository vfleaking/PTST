B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

LLAMA_BASE_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
LLAMA_SHORT_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
MPT_PROMPT = "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."

STYLE_TRANSLATION_MAP = {
    'gsm:text:vanilla': 'text_question_answer',
    'gsm:text:alpaca': 'text_alpaca',
    'gsm:chat:vanilla': 'question',
    'gsm:chat:alpaca': 'question_alpaca',
    'gsm:chat:llama': 'question_llama',
    'gsm:chat:llama-short': 'question_llama_short',
    'gsm:chat:mpt': 'question_mpt',
}

def get_prompt_template(prompt_template_style="chat"):
    add_inst = True
    response_prefix = None

    if prompt_template_style in STYLE_TRANSLATION_MAP:
        prompt_template_style = STYLE_TRANSLATION_MAP[prompt_template_style]
    
    if prompt_template_style == 'text':
        PROMPT_TEMPLATE = "%s"
        add_inst = False
    elif prompt_template_style == 'chat':
        PROMPT_TEMPLATE = "%s"
    elif prompt_template_style == 'text_question_answer':
        PROMPT_TEMPLATE = "Question: %s\nAnswer:"
        add_inst = False
    elif prompt_template_style == 'text_alpaca':
        PROMPT_TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n" \
            + "### Instruction:\n%s\n\n### Response:\n"
        add_inst = False
    elif prompt_template_style == 'question':
        PROMPT_TEMPLATE = "Question: %s"
    elif prompt_template_style == 'question_alpaca':
        PROMPT_TEMPLATE = B_SYS + "Below is an instruction that describes a task. Write a response that appropriately completes the request." + E_SYS \
            + "### Instruction:\n%s\n\n### Response:\n"
    elif prompt_template_style == 'question_llama':
        PROMPT_TEMPLATE = B_SYS + LLAMA_BASE_PROMPT + E_SYS + "Question: %s"
    elif prompt_template_style == 'question_llama_short':
        PROMPT_TEMPLATE = B_SYS + LLAMA_SHORT_PROMPT + E_SYS + "Question: %s"
    elif prompt_template_style == 'question_mpt':
        PROMPT_TEMPLATE = B_SYS + MPT_PROMPT + E_SYS + "Question: %s"
    else:
        raise ValueError("Invalid prompt template style.")
    
    if add_inst:
        PROMPT_TEMPLATE = B_INST + " " + PROMPT_TEMPLATE + " " + E_INST
    
    if response_prefix is not None:
        PROMPT_TEMPLATE = PROMPT_TEMPLATE + " " + response_prefix
    
    return PROMPT_TEMPLATE


def apply_prompt_template(prompt_template_style='base', dataset=None, tokenizer=None, prefix="", return_dialogs=False):
    
    # Retrieve the prompt template
    PROMPT_TEMPLATE = get_prompt_template(prompt_template_style)

    # Save and tokenize every dialog!
    dialogs = []
    chats = []
    
    for prompt in dataset:
        prompt = PROMPT_TEMPLATE % (prefix + prompt)
        dialogs.append(prompt)
        chats.append(tokenizer.encode(prompt))
    
    if return_dialogs:
        return chats, dialogs
    else:
        return chats
