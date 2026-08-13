import time
from langsmith import traceable, get_current_run_tree
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


from app.rag.providers import LLMProvider, get_provider


class LLMService:
    """Generates answers using the configured LLM provider."""

    def __init__(
        self,
        provider: str | LLMProvider = "ollama",
        model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        api_key: str = "",
    ):
        if isinstance(provider, LLMProvider):
            self.provider_instance = provider
            self.provider_name = provider.provider_name
            self.model_name = provider.model_name
        else:
            self.provider_name = provider
            self.model_name = model_name
            self.provider_instance = get_provider(
                provider_name=provider,
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
            )

    @traceable(name="llm_generation", run_type="llm")
    def generate(self, question: str, context: list[str]) -> str:
        start_time = time.time()
        if not context:
            answer = "I cannot find the answer in the provided context."
            elapsed = time.time() - start_time
            try:
                rt = get_current_run_tree()
                if rt:
                    rt.metadata["answer_generator_provider"] = self.provider_name
                    rt.metadata["answer_generator_model"] = self.model_name
                    rt.metadata["model_name"] = self.model_name
                    rt.metadata["context_chunk_count"] = 0
                    rt.metadata["answer_length"] = len(answer)
                    rt.metadata["generation_latency"] = round(elapsed, 4)
            except Exception:
                pass
            return answer

        context_str = "\n\n---\n\n".join(context)
        prompt = (
            "Answer the question based ONLY on the supplied context.\n"
            "If the answer is not supported by the context, state clearly: "
            "\"I cannot find the answer in the provided context.\"\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        res = self.provider_instance.generate(prompt, temperature=0.1)
        if isinstance(res, dict):
            if res.get("success"):
                answer = (res.get("text") or res.get("content") or "").strip()
            else:
                answer = f"Error from LLM Provider ({self.provider_name}): {res.get('error')}"
        else:
            answer = str(res).strip()

        elapsed = time.time() - start_time

        try:
            rt = get_current_run_tree()
            if rt:
                rt.metadata["answer_generator_provider"] = self.provider_name
                rt.metadata["answer_generator_model"] = self.model_name
                rt.metadata["model_name"] = self.model_name
                rt.metadata["context_chunk_count"] = len(context)
                rt.metadata["answer_length"] = len(answer)
                rt.metadata["generation_latency"] = round(elapsed, 4)
        except Exception:
            pass

        return answer

