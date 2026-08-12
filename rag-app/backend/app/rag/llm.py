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

    def generate(self, question: str, context: list[str], chat_history: list = None) -> str:
        if not context:
            return "I cannot find the answer in the provided context."

        context_str = "\n\n---\n\n".join(context)
        history_str = ""
        if chat_history:
            formatted_msgs = []
            for msg in chat_history:
                role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                if role and content:
                    role_label = "User" if role == "user" else "Assistant"
                    formatted_msgs.append(f"{role_label}: {content}")
            if formatted_msgs:
                history_str = "Chat History:\n" + "\n".join(formatted_msgs) + "\n\n"

        prompt = (
            "Answer the question based ONLY on the supplied context.\n"
            "If the answer is not supported by the context, state clearly: "
            "\"I cannot find the answer in the provided context.\"\n\n"
            f"{history_str}"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        res = self.llm.invoke(prompt)
        ans = res.content if hasattr(res, "content") and res.content is not None else str(res)
        if isinstance(ans, list):
            ans = "".join([str(x) for x in ans])
        return str(ans).strip()

    def generate_conversational(self, question: str, chat_history: list = None) -> str:
        history_str = ""
        if chat_history:
            formatted_msgs = []
            for msg in chat_history:
                role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                if role and content:
                    role_label = "User" if role == "user" else "Assistant"
                    formatted_msgs.append(f"{role_label}: {content}")
            if formatted_msgs:
                history_str = "Chat History:\n" + "\n".join(formatted_msgs) + "\n\n"

        prompt = (
            "You are a helpful assistant.\n"
            "Answer the user's question directly using the conversation history if relevant.\n"
            "Pay close attention to any user details (such as names, preferences, or previous statements) mentioned in the Chat History.\n\n"
            f"{history_str}"
            f"User Question: {question}\n\nAnswer:"
        )
        res = self.llm.invoke(prompt)
        ans = res.content if hasattr(res, "content") and res.content is not None else str(res)
        if isinstance(ans, list):
            ans = "".join([str(x) for x in ans])
        return str(ans).strip()
