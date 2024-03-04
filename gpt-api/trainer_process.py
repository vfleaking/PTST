import wandb
import time
import csv
import os
import json
import io
import requests
import pandas as pd
class TrainerProcess:
    def __init__(self, configs, openai_cli):
        self.configs = configs
        self.openai_cli = openai_cli

    def upload_file(self, fname):
        uploaded_file = self.openai_cli.files.create(
            file=open(fname, "rb"),
            purpose="fine-tune",
        )
        return uploaded_file

    def wait_for_file_processing(self, uploaded_file, sleep_time=5):
        while uploaded_file.status != 'processed':
            time.sleep(sleep_time)
            uploaded_file.refresh()
            print(f"File Status: {uploaded_file.status} " )

    
    def wait_for_ft(self, job_id, sleep_time=10):
        ft_job = self.openai_cli.fine_tuning.jobs.retrieve(job_id)
        while ft_job.status != 'succeeded':
            time.sleep(sleep_time)
            ft_job = self.openai_cli.fine_tuning.jobs.retrieve(job_id)
            print(f"Fine-tuning Job Status: {ft_job.status}")
            if ft_job.error != None:
                print(f"Fine-tuning Job Error: {ft_job.error}")
            if ft_job.result_files: 
                print("Fine-tuning Job Result Files: ", ft_job.result_files)
        
        print(f"Ft job after completion {ft_job}")
        print("Fine-tuning Job Result Files: ", ft_job.result_files)
        assert len(ft_job.result_files) == 1, "result files not equal to 1"
        result_file_id = ft_job.result_files[0]
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        try:
            response = requests.get(f"https://api.openai.com/v1/files/{result_file_id}/content", headers=headers)
            data_io = io.StringIO(response.text)
            # df = pd.read_csv(data_io)
            # print(df)
            reader = csv.reader(data_io)
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                wandb_dict = {
                    'train_step': int(row[0]),
                    'train_loss': float(row[1]),
                    'train_acc': float(row[2]),
                    }
                if row[3]:
                    wandb_dict['val_loss'] = float(row[3])
                if row[4]:
                    wandb_dict['val_mean_token_acc'] = float(row[4])
                wandb.log(wandb_dict, commit=True)


        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as err:
            print(f"An error occurred: {err}")




        ft_model_id =  ft_job.fine_tuned_model
        wandb.config.ft_model_id = ft_model_id
        print(f'\n fine-tuned model id: {ft_model_id} \n')





    def run(self, fname, suffix, job_id=None, sleep_time=10):
        if not job_id:
            uploaded_file = self.upload_file(fname)
            print(f'file id: {uploaded_file.id}')
            self.wait_for_file_processing(uploaded_file)
            hyperparam_dict = {"n_epochs": self.configs['n_epochs']}
            if self.configs['batch_size']:
                hyperparam_dict['batch_size'] = self.configs['batch_size']
            ft_job = self.openai_cli.fine_tuning.jobs.create(
                training_file=uploaded_file.id,  
                model=self.configs['model'],
                hyperparameters=hyperparam_dict,
                suffix=suffix
                )
            print(f'fine-tuning job id: {ft_job.id}')
            job_id = ft_job.id
            wandb.config.job_id = job_id
        else:
            # job_id = 'ftjob-p9QWbxPL0rbH3e35qxw0gC7E'
            ft_job = self.openai_cli.fine_tuning.jobs.retrieve(job_id)

        print(f'\n ft job {ft_job} \n')
        
        self.wait_for_ft(job_id, sleep_time=sleep_time)








