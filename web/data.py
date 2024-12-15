from dataclasses import dataclass
from pathlib import Path
import shutil
import ollama
from openai import OpenAI
import streamlit as st
from huggingface_hub import HfApi
import json

@dataclass
class Model:
    display_name: str
    model_name: str

@st.cache_data
def get_available_models():
    print(st.session_state)
    return {
        "[Local] Ollama": [Model(display_name=f"{model.model} [{model.details.parameter_size}]", model_name=model.model)
                           for model in ollama.list().models],
        "[Cloud] OpenAI": [Model(display_name=model.id, model_name=model.id)
                           for model in OpenAI(api_key=st.session_state.openai_key).models.list()
                           ] if hasattr(st.session_state, 'openai_key') and st.session_state.openai_key is not None else []
    }

DATASETS_DIR = Path(__file__).parent.parent / "datasets"

@st.cache_data
def get_available_datasets():
    return {f.name: f for f in DATASETS_DIR.iterdir() if f.is_dir() and f.name != "default"}

def remove_dataset(name):
    shutil.rmtree(DATASETS_DIR.joinpath(name))

def create_dataset(name):
    dataset_path = DATASETS_DIR.joinpath(name)
    dataset_path.mkdir(parents=False, exist_ok=False)
    return dataset_path

PROMPTS_FILE = Path(__file__).parent.parent.joinpath('prompts.jsonl')

@st.cache_data
def get_available_prompts():
    with PROMPTS_FILE.open() as f:
        return [
            json.loads(line)
            for line in f.readlines()
        ]

def save_prompts(data):
    with PROMPTS_FILE.open('w') as f:
        for item in data:
            f.write(json.dumps(item))
            f.write('\n')

def search_embedding_models(query=None, limit=5):
    return [model.id for model in HfApi().list_models(
        task="sentence-transformers",
        limit=limit,
        search=query
    )]