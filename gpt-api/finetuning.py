import wandb
from openai import OpenAI
import fire
import os
import data_utils
from typing import List, Union
import copy
from wandb.integration.openai import autolog
from trainer_process import TrainerProcess
import datasets
# openai.api_key = os.environ["OPENAI_API_KEY"]
PROJ_NAME = 'llm-safety'
WANDB_ENTITY = 'gxr-team'
os.environ["WANDB__SERVICE_WAIT"] = "300"

trigger_sync = None

def set_wandb(config):

    if config['wandb'] == 'offline-sync':
        mode = 'offline'
        print('offline mode. wandb osh enabled')
        from wandb_osh.hooks import TriggerWandbSyncHook
        global trigger_sync
        trigger_sync = TriggerWandbSyncHook()
    else:
        mode =config['wandb']
    data_basename, _ = os.path.splitext(os.path.basename(config['data']))
    wandb_name = f'data={data_basename}-epochs={config["n_epochs"]}'
    config['wandb_name'] = wandb_name
    # wandb_dict = {
    #     'mode': mode,
    #     'project': PROJ_NAME,
    #     'entity': WANDB_ENTITY,
    #     'name': wandb_name,
    #     'config': config,
    #     'settings': wandb.Settings(_service_wait=300),
    # }
    # autolog(wandb_dict)

    run = wandb.init(
        mode=mode,
        project=PROJ_NAME,
        entity=WANDB_ENTITY,
        name=wandb_name,
        config=config,
        settings=wandb.Settings(_service_wait=300),
    )
    wandb.run.log_code(".")


def main(
        data: str = 'gsm8k',
        use_hf_data: bool = True,
        hf_data_split: str = 'train',
        data_prompt_key: str = 'prompt',
        data_answer_key: str = 'answer',
        template: str = 'vanilla',
        n_epochs=10,
        batch_size=None,
        seed=0,
        sys_pdir = './sys_prompts',
        model = 'gpt-3.5-turbo-0613',
        wandb: str = 'online',
        job_id: str = None,
        max_retries: int = 5,
        ):

    
    config_dict = copy.deepcopy(locals())

    if job_id is None:
        config_dict.pop('job_id')

    

    set_wandb(config_dict)

    dt_fname = data_utils.gen_data(data=data, n_epochs=n_epochs, use_hf_data=use_hf_data, prompt_key=data_prompt_key, answer_key=data_answer_key, template=template)


    

    openai_cli = OpenAI()
    tp = TrainerProcess(config_dict, openai_cli)
    tp.run(dt_fname, suffix=f'seed={seed}', job_id=job_id)



if __name__ == "__main__":
    fire.Fire(main)

