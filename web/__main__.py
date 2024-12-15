import os
from types import SimpleNamespace

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from agent import AgentBuilder
from web.data import get_available_models, get_available_prompts, get_available_datasets
from web.modals import configure_models, configure_prompts, configure_datasets
import streamlit as st

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

st.set_page_config(
    page_title="LangGraph Agent Demo",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

available_models = get_available_models()
available_prompts = get_available_prompts()
available_datasets = get_available_datasets()

if "llm_config" not in st.session_state or \
        any(map(lambda curr_dataset: curr_dataset not in available_datasets, st.session_state.llm_config.datasets)) or\
        st.session_state.llm_config.system_prompt not in available_prompts or\
        st.session_state.llm_config.provider not in available_models or\
        st.session_state.llm_config.model not in available_models[st.session_state.llm_config.provider]:
    current_datasets = [] if "llm_config" not in st.session_state else (
        list(filter(lambda curr_dataset: curr_dataset in available_datasets, st.session_state.llm_config.datasets)))
    st.session_state.llm_config = SimpleNamespace(
        provider=list(available_models.keys())[0],
        model=list(available_models.values())[0][0],
        system_prompt=next(iter(available_prompts), None),
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        datasets=current_datasets,
        temperature=0,
        openai_key=None,
    )

@st.cache_resource
def get_agent():
    if st.session_state.llm_config.provider == "[Cloud] OpenAI":
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
        model = ChatOpenAI(model_name=st.session_state.llm_config.model.model_name,
                       temperature=st.session_state.llm_config.temperature)
    else:
        model = ChatOllama(model=st.session_state.llm_config.model.model_name,
                           temperature=st.session_state.llm_config.temperature,
                           format='')

    return AgentBuilder(
        model=model,
        prompt=st.session_state.llm_config.system_prompt or "You are a helpful assistant",
        embedding_model_name=st.session_state.llm_config.embedding_model_name,
        dataset_dirs=[available_datasets[dataset_name] for dataset_name in st.session_state.llm_config.datasets],
    ).build()

agent = get_agent()

st.title("Chat with agent!")
st.markdown(f"Writing to: ⭐️{agent.name}⭐️")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.session_state.llm_config.datasets = st.sidebar.pills(
    "Select datasets",
    list(available_datasets.keys()),
    default=st.session_state.llm_config.datasets,
    selection_mode="multi",
)

st.session_state.llm_config.system_prompt = st.sidebar.radio(
    "Select system prompt",
    available_prompts,
    index=available_prompts.index(st.session_state.llm_config.system_prompt) if st.session_state.llm_config.system_prompt else None,
    format_func=lambda prompt: f"{prompt[:30]}..." if len(prompt) > 30 else prompt,
)

# Using object notation
st.session_state.llm_config.provider = st.sidebar.selectbox(
    "LLM Provider",
    (available_models.keys())
)

if st.session_state.llm_config.provider == "[Cloud] OpenAI":
    st.sidebar.text_input("API KEY: ", key="openai_key", on_change=lambda: st.cache_data.clear())
else:
    st.session_state.openai_key = None

st.session_state.llm_config.model = st.sidebar.radio(
    "Select model",
    available_models[st.session_state.llm_config.provider],
    index=available_models[st.session_state.llm_config.provider].index(st.session_state.llm_config.model)
    if st.session_state.llm_config.model in available_models[st.session_state.llm_config.provider] else None,
    format_func=lambda model: model.display_name,
)

st.session_state.llm_config.temperature = st.sidebar.slider("LLM Temperature: ", 0.0, 1.0)

if st.sidebar.button("Send Config", icon=":material/send:", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

if st.sidebar.button("Configure Models", icon=":material/settings:", use_container_width=True):
    configure_models(provider=st.session_state.llm_config.provider)

if st.sidebar.button("Configure Prompts", icon=":material/chat:", use_container_width=True):
    configure_prompts()

if st.sidebar.button("Configure Datasets", icon=":material/storage:", use_container_width=True):
    configure_datasets()

if prompt := st.chat_input("Ask your agent something..."):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    answer = agent.ask(prompt)
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})