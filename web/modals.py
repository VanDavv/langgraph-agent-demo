import streamlit as st
from web.data import get_available_models, get_available_prompts, save_prompts, get_available_datasets, remove_dataset, \
    create_dataset, search_embedding_models
import ollama

@st.dialog("Configure Datasets")
def configure_datasets():
    st.write(f"Current embedding model: {st.session_state.llm_config.embedding_model_name}")
    name = st.text_input("Search embedding models")

    def update():
        st.session_state.llm_config.embedding_model_name = st.session_state['new_embedding_model']

    st.selectbox(
        "Select new model",
        search_embedding_models(name),
        key="new_embedding_model",
        index=None,
        on_change=update,
    )

    datasets = get_available_datasets()
    selection = st.pills("Select datasets", datasets, selection_mode="multi", key="configure_datasets_selection")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Delete", key="configure_datasets_delete"):
            for selected in selection:
                remove_dataset(selected)

            if any(filter(lambda item: item in st.session_state.llm_config.datasets, selection)):
                st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    with col2:
        with st.popover("Add"):
            name = st.text_input("Dataset name")
            uploaded_files = st.file_uploader(
                "Choose dataset files", accept_multiple_files=True
            )
            if st.button("Run", key="configure_datasets_run"):
                with st.spinner('Running...'):
                    dataset_path = create_dataset(name)
                    for uploaded_file in uploaded_files:
                        with dataset_path.joinpath(uploaded_file.name).open('wb') as file:
                            file.write(uploaded_file.read())
                st.cache_data.clear()
                st.rerun()

@st.dialog("Configure Models")
def configure_models(provider):
    st.write(f"Configuring {provider} models")
    models = get_available_models()[provider]
    selection = st.pills("Select models", models, selection_mode="multi", format_func=lambda model: model.display_name)

    if provider == "[Local] Ollama":
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Delete"):
                for selected in selection:
                    ollama.delete(selected.model_name)
                    if selected == st.session_state.llm_config.model:
                        st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()

        with col2:
            with st.popover("Download"):
                name = st.text_input("Model name")
                if st.button("Run"):
                    with st.spinner('Downloading...'):
                        ollama.pull(name)
                    st.cache_data.clear()
                    st.rerun()

@st.dialog("Configure System Prompts")
def configure_prompts():
    prompts = get_available_prompts()
    selection = st.pills("Select prompts", prompts, selection_mode="multi",
                         format_func=lambda prompt: f"{prompt[:30]}..." if len(prompt) > 30 else prompt)

    if st.button("Delete", use_container_width=True):
        save_prompts([prompt for prompt in prompts if prompt not in selection])
        if any(filter(lambda selected: selected == st.session_state.llm_config.system_prompt, selection)):
            st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    if len(selection) == 1:
        with st.popover("Edit", use_container_width=True):
            new_prompt = st.text_area("Edited prompt", selection[0])
            if st.button("Save", key="save_edited_prompt"):
                save_prompts([prompt for prompt in prompts if prompt not in selection] + [new_prompt])
                if any(filter(lambda selected: selected == st.session_state.llm_config.system_prompt, selection)):
                    st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()

    with st.popover("Create", use_container_width=True):
        new_prompt = st.text_area("New prompt")
        if st.button("Save", key="save_new_prompt"):
            save_prompts(prompts + [new_prompt])
            st.cache_data.clear()
            st.rerun()
