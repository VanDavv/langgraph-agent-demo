from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from store import VectorStore


class Agent:
    def __init__(self, name, graph):
        self.name = name
        self.graph = graph

    def ask(self, query):
        config = {"configurable": {"thread_id": "1"}}
        response = ""
        events = self.graph.stream({"question": query}, config, stream_mode="values")
        for event in events:
            if event.get("stage", None) == "chatbot" and event["question"] == query:
                response = event["answer"]
        return response


class AgentBuilder:
    def __init__(self, model: ChatOllama | ChatOpenAI, prompt: str, dataset_dirs: list = None, embedding_model_name=None):
        self.llm = model
        self.prompt = prompt
        self.nodes = {
            "clean": self.clean,
            "chatbot": self.chatbot,
            "retrieve": self.retrieve,
        }
        self.connections = [
            (START, "clean"),
            ("clean", "retrieve"),
            ("retrieve", "chatbot"),
            ("chatbot", END),
        ]

        import time
        start = time.time()
        self._store = VectorStore(dataset_dirs, embedding_model=embedding_model_name, force_reload=True)
        end = time.time()
        print(f"Vector store loading time: {end - start:.3f} seconds")

    def build(self):
        graph_builder = StateGraph(AgentBuilder.GraphState)
        memory = MemorySaver()

        for node_name, node_fn in self.nodes.items():
            graph_builder.add_node(node_name, node_fn)

        for from_node, to_node in self.connections:
            graph_builder.add_edge(from_node, to_node)

        if isinstance(self.llm, ChatOpenAI):
            name = self.llm.model_name
        else:
            name = self.llm.model
        return Agent(name=name, graph=graph_builder.compile(checkpointer=memory))

    class GraphState(TypedDict):
        messages: Annotated[list, add_messages]
        question: str
        answer: str
        stage: str

    def chatbot(self, state: "AgentBuilder.GraphState"):
        return {**state, "answer": self.llm.invoke(state["messages"]).content, "stage": "chatbot"}

    def retrieve(self, state: "AgentBuilder.GraphState"):
        context = '\n\n'.join([doc.page_content for doc in self._store.retrieve(state["question"])])

        messages = [
            ("system", self.prompt.format(context=context)),
            ("user", state["question"])
        ]
        return {**state, "stage": "retrieve", "messages": messages}

    def clean(self, state: "AgentBuilder.GraphState"):
        return {"answer": None, "context": None, "messages": [], "question": state["question"], "stage": "clean"}