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
            "You are a precise technical assistant for a RAG application.\n"
            "Answer the user's question directly using ONLY the facts present in the supplied Context.\n\n"
            "STRICT RULES:\n"
            "1. Primary Factual Source: Base your response exclusively on the supplied Context. Do not invent or assume unsupported facts.\n"
            "2. Insufficient Context: If the supplied context does not contain enough information to answer the question, state clearly:\n"
            '   "I cannot find the answer in the provided context."\n'
            "3. Direct Answer: Answer the user's question directly without preamble.\n"
            "4. Internal Systems: Never mention internal processes such as retrieval, vector search, chunks, or reranking.\n"
            "5. Technical Precision: Preserve all relevant mathematical notation, code snippet syntax, and technical terminology exactly.\n"
            "6. Source Distinction: When information comes from different documents in the context, explicitly distinguish the sources.\n\n"
            f"{history_str}"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
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
            "You are a helpful AI assistant.\n"
            "Answer the user's input directly and concisely.\n"
            "If the Chat History contains user details (such as names, preferences, or past statements), use them accurately.\n\n"
            f"{history_str}"
            f"User Question: {question}\n\nAnswer:"
        )
        res = self.llm.invoke(prompt)
        ans = res.content if hasattr(res, "content") and res.content is not None else str(res)
        if isinstance(ans, list):
            ans = "".join([str(x) for x in ans])
        return str(ans).strip()

    def generate_summary(self, doc_name: str, context: list[str]) -> str:
        if not context:
            return f"No content available to summarize for '{doc_name}'."

        batch_size = 4
        if len(context) > batch_size:
            batch_summaries = []
            for i in range(0, len(context), batch_size):
                batch_chunks = context[i:i + batch_size]
                batch_text = "\n\n---\n\n".join(batch_chunks)
                b_prompt = (
                    f"Summarize section {i // batch_size + 1} of document '{doc_name}' based strictly on the text below:\n"
                    f"- Extract core technical concepts and structural headings.\n"
                    f"- Do not add external information or opinion.\n\n"
                    f"Text:\n{batch_text}\n\nSection Summary:"
                )
                res = self.llm.invoke(b_prompt)
                ans = res.content if hasattr(res, "content") and res.content is not None else str(res)
                batch_summaries.append(str(ans).strip())

            combined_text = "\n\n".join(batch_summaries)
            final_prompt = (
                f"Synthesize the following section summaries into a single, cohesive summary of '{doc_name}':\n"
                "STRICT RULES:\n"
                "- Maintain the document's original logical structure and key terminology.\n"
                "- Do not introduce external knowledge or facts not present in the summaries.\n"
                "- Produce a fluid, unified document summary rather than a list of disconnected section notes.\n\n"
                f"Section Summaries:\n{combined_text}\n\nFinal Summary:"
            )
            res = self.llm.invoke(final_prompt)
            ans = res.content if hasattr(res, "content") and res.content is not None else str(res)
            return str(ans).strip()

        context_str = "\n\n---\n\n".join(context)
        prompt = (
            f"Provide a concise, faithful summary of the document '{doc_name}' based ONLY on the supplied content:\n"
            "STRICT RULES:\n"
            "- Preserve core technical concepts, mathematical formulas, and logical structure.\n"
            "- Do not introduce external facts or commentary.\n"
            "- Be direct and well-structured.\n\n"
            f"Document Content:\n{context_str}\n\nSummary:"
        )
        res = self.llm.invoke(prompt)
        ans = res.content if hasattr(res, "content") and res.content is not None else str(res)
        return str(ans).strip()

    def generate_comparison(self, doc_a: str, context_a: list[str], doc_b: str, context_b: list[str], question: str) -> str:
        text_a = "\n\n".join(context_a) if context_a else "No content retrieved from " + doc_a
        text_b = "\n\n".join(context_b) if context_b else "No content retrieved from " + doc_b

        prompt = (
            f"Compare the information in '{doc_a}' and '{doc_b}' concerning: '{question}'.\n\n"
            "STRICT RULES:\n"
            "1. Source Attribution: Keep evidence from each document distinct. Clearly attribute claims to their respective source.\n"
            "2. Honest Comparison: Identify key similarities and differences supported by the context.\n"
            "3. Fact Preservation: Do not attribute facts to the wrong document. Do not invent differences or facts if evidence is missing.\n"
            "4. Technical Precision: Retain exact terminology and notation.\n\n"
            f"--- Evidence from {doc_a} ---\n{text_a}\n\n"
            f"--- Evidence from {doc_b} ---\n{text_b}\n\n"
            "Comparison:"
        )
        res = self.llm.invoke(prompt)
        ans = res.content if hasattr(res, "content") and res.content is not None else str(res)
        return str(ans).strip()


