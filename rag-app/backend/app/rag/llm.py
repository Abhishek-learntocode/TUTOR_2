from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from app.models.state import Message


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

    def _format_history(self, chat_history: list[Message] = None) -> str:
        if not chat_history:
            return ""
        lines = []
        for msg in chat_history[-6:]:
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\nConversation History:\n" + "\n".join(lines) + "\n\n"

    def generate(self, question: str, context: list[str], chat_history: list[Message] = None) -> str:
        if not context:
            return "I cannot find the answer in the provided context."

        context_str = "\n\n---\n\n".join(context)
        history_str = self._format_history(chat_history)
        prompt = (
            "You are a helpful assistant. Answer the user question based ONLY on the supplied context and conversation history.\n"
            "If the answer is not supported by the context, state clearly: "
            "\"I cannot find the answer in the provided context.\"\n\n"
            f"{history_str}"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        res = self.llm.invoke(prompt)
        return res.content.strip() if hasattr(res, "content") else str(res).strip()

    def generate_conversational(self, question: str, chat_history: list[Message] = None) -> str:
        history_str = self._format_history(chat_history)
        prompt = (
            "You are a friendly, intelligent assistant engaged in a conversation.\n"
            "Respond naturally to the user question using the conversation history when relevant.\n"
            "If the user asks for their name or details they previously shared, use the conversation history to answer accurately.\n\n"
            f"{history_str}"
            f"User Question: {question}\n\nResponse:"
        )
        res = self.llm.invoke(prompt)
        return res.content.strip() if hasattr(res, "content") else str(res).strip()

    def generate_summary(self, doc_id: str, chunks: list[str]) -> str:
        context_str = "\n\n---\n\n".join(chunks[:6])
        prompt = (
            f"Summarize the document '{doc_id}' concisely based ONLY on the provided context.\n\n"
            f"Context:\n{context_str}\n\nSummary:"
        )
        res = self.llm.invoke(prompt)
        return res.content.strip() if hasattr(res, "content") else str(res).strip()

    def generate_comparison(
        self, doc_a: str, chunks_a: list[str], doc_b: str, chunks_b: list[str], question: str
    ) -> str:
        ctx_a = "\n\n".join(chunks_a[:3])
        ctx_b = "\n\n".join(chunks_b[:3])
        prompt = (
            f"Compare document '{doc_a}' with document '{doc_b}' regarding the question: '{question}'.\n\n"
            f"Document A ('{doc_a}') Context:\n{ctx_a}\n\n"
            f"Document B ('{doc_b}') Context:\n{ctx_b}\n\n"
            "Comparison:"
        )
        res = self.llm.invoke(prompt)
        return res.content.strip() if hasattr(res, "content") else str(res).strip()
