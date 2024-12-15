from pathlib import Path

import chromadb
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

DEFAULT_PERSIST_PATH = Path(__file__).parent.absolute() / 'vector_store'

class VectorStore:
    def __init__(self, input_dirs: list[Path] = None, embedding_model="BAAI/bge-m3", persist_dir: Path = DEFAULT_PERSIST_PATH, force_reload=False):
        if input_dirs is None or len(input_dirs) == 0:
            input_dirs = [Path(__file__).parent.absolute() / 'datasets' / 'default']
        self._input_dirs = input_dirs
        self._persist_dir = persist_dir
        self._force_reload = force_reload
        self._embedding_model = HuggingFaceBgeEmbeddings(
            # Jest >9500 modeli https://huggingface.co/models?library=sentence-transformers&sort=downloads
            # model_name="BAAI/bge-small-en",
            # model_name="uaritm/multilingual_en_uk_pl_ru",
            model_name=embedding_model,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self._store = self._prepare().as_retriever()

    def _get_documents(self):
        documents = []
        for input_dir in self._input_dirs:
            for file in input_dir.rglob('*.pdf'):
                documents.extend(PyPDFLoader(file).load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_documents(documents)

    def _prepare(self) -> Chroma:
        if self._force_reload:
            if self._persist_dir.exists():
                import shutil
                shutil.rmtree(self._persist_dir)
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        if self._persist_dir.exists():
            return Chroma(embedding_function=self._embedding_model, persist_directory=str(self._persist_dir))
        else:
            return Chroma.from_documents(documents=self._get_documents(), embedding=self._embedding_model,
                                         persist_directory=str(self._persist_dir))

    def retrieve(self, query):
        return self._store.invoke(query)