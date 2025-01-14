import os
import sys
from typing import Union, OrderedDict
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder
# Load the .env file if it exists
load_dotenv()

# Hardcoded path to config file
CONFIG_PATH = "/workspace/ai-toolkit/config/examples/train_lora_flux_24gb.yaml"

# Insert the directory path into sys.path
sys.path.insert(0, os.path.dirname(CONFIG_PATH))

# turn off diffusers telemetry
os.environ['DISABLE_TELEMETRY'] = 'YES'

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)

from toolkit.job import get_job

def print_end_message(jobs_completed, jobs_failed):
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print("")
    print("========================================")
    print("Result:")
    if len(completed_string) > 0:
        print(f" - {completed_string}")
    if len(failure_string) > 0:
        print(f" - {failure_string}")
    print("========================================")

def set_hf_token(token: str):
    """Set Hugging Face token in both environment and HfFolder"""
    os.environ["HF_TOKEN"] = token
    HfFolder.save_token(token)
    print(f"Set new Hugging Face token: {token[:6]}...")

def main():
    jobs_completed = 0
    jobs_failed = 0

    print(f"Running job with config: {CONFIG_PATH}")

    try:
        # Load first token from .env
        first_token = os.getenv("HF_TOKEN_FIRST")
        if not first_token:
            raise ValueError("HF_TOKEN_FIRST not found in .env file")
        set_hf_token(first_token)

        job = get_job(CONFIG_PATH, name=None)
        job.run()
        job.cleanup()
        jobs_completed += 1

        # Load second token from .env
        second_token = os.getenv("HF_TOKEN_SECOND")
        if not second_token:
            raise ValueError("HF_TOKEN_SECOND not found in .env file")
        set_hf_token(second_token)

        repo_id =  os.getenv("REPO_ID")

        # Upload your LoRA weights
        api = HfApi()
        api.upload_file(
            path_or_fileobj="/workspace/ai-toolkit/output/azizshaw007/azizshaw007.safetensors",
            path_in_repo="azizshaw007.safetensors",
            repo_id=repo_id,
            repo_type="model"
        )

        print(f"LoRA weights pushed to: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"Error running job: {e}")
        jobs_failed += 1
        print_end_message(jobs_completed, jobs_failed)
        raise e

    print_end_message(jobs_completed, jobs_failed)

if __name__ == '__main__':
    main()
