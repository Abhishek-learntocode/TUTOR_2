from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class LLMService:
    """Generates answers using the configured LLM."""

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        api_key: str = "",
    ):
        if provider == "openai":
            self.llm = ChatOpenAI(model=model_name, api_key=api_key or None, temperature=0.1)
        else:
            self.llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.1)

    def generate(self, question: str, context: list[str]) -> str:
        if not context:
            return "I cannot find the answer in the provided context."

        context_str = "\n\n---\n\n".join(context)
        prompt = (
            "Answer the question based ONLY on the supplied context.\n"
            "If the answer is not supported by the context, state clearly: "
            "\"I cannot find the answer in the provided context.\"\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        res = self.llm.invoke(prompt)
        return res.content.strip() if hasattr(res, "content") else str(res).strip()
