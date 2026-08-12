import json
import re
from app.models.state import QueryAnalysis, Message


def is_conversational_query(query: str) -> bool:
    """Detect if a query is conversational (user name, greeting, chat history question, small talk)."""
    q_lower = query.lower().strip()
    conversational_patterns = [
        r"\b(my name|who am i|what is my name|remember my name|do you know my name|call me|i am \w+)\b",
        r"\b(hello|hi|hey|good morning|good afternoon|good evening|how are you|who are you|what can you do|thank you|thanks)\b",
        r"\b(what did i ask|what was my previous|repeat what i said|what was my first question)\b",
    ]
    for pattern in conversational_patterns:
        if re.search(pattern, q_lower):
            return True
    return False


def resolve_reference_fallback(query: str, chat_history: list) -> str:
    """Fallback reference resolution if LLM failed to resolve pronouns like 'its', 'it', 'this', 'that'."""
    pronoun_pattern = r"\b(its|it|this|that|these|those|their)\b"
    if not re.search(pronoun_pattern, query, re.I) or not chat_history:
        return query

    last_user_turn = ""
    for msg in reversed(chat_history):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
        if role == "user":
            if not is_conversational_query(content):
                last_user_turn = content
                break

    if not last_user_turn:
        return query

    clean = re.sub(
        r"^(what|how|why|where|when|who|is|are|can|could|explain|describe|tell me about)\s+(is|are|a|an|the)?\s*",
        "",
        last_user_turn,
        flags=re.I,
    )
    clean = clean.strip("? .!\n")

    if clean:
        resolved = re.sub(pronoun_pattern, f"the {clean}", query, count=1, flags=re.I)
        return resolved
    return query


class QueryAnalyzer:
    """Analyzes queries using LLM + rule heuristics to classify as single_hop, multi_hop, or conversational."""

    def __init__(self, llm):
        self.llm = llm

    def analyze(self, query: str, chat_history: list[Message] = None) -> QueryAnalysis:
        if is_conversational_query(query):
            return QueryAnalysis(query_type="conversational", sub_queries=[query])

        query_lower = query.lower()
        has_multi_hop_cue = (
            "compare" in query_lower
            or "difference between" in query_lower
            or " versus " in query_lower
            or " vs " in query_lower
            or bool(re.search(r"chapter\s+\d+.*chapter\s+\d+", query, re.I))
        )

        history_text = ""
        if chat_history:
            formatted = []
            for msg in chat_history:
                role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                if role and content:
                    role_label = "User" if role == "user" else "Assistant"
                    formatted.append(f"{role_label}: {content}")
            if formatted:
                history_text = "Chat History:\n" + "\n".join(formatted) + "\n\n"

        prompt = (
            "You are a Query Analyzer for a RAG system.\n"
            "Analyze the current user question in light of the chat history (if provided).\n"
            "Your tasks are:\n"
            "1. Reference Resolution: Rewrite follow-up questions containing pronouns or implicit references "
            "(e.g., 'its', 'it', 'this', 'that', 'the former', 'how does it work', 'what are its advantages') "
            "into self-contained, standalone search queries using the chat history.\n"
            "2. Classification: Classify the query as 'single_hop' or 'multi_hop' and generate 'sub_queries'.\n\n"
            "RULES:\n"
            "1. Default to 'single_hop' for questions asking about a single concept, definition, or direct topic.\n"
            "2. Classify as 'multi_hop' ONLY when answering requires combining or comparing separate topics/chapters.\n"
            "3. For single_hop, sub_queries must contain ONLY the single standalone question (with references resolved if chat history is present).\n"
            "4. For multi_hop, generate 2-3 focused standalone sub-queries for each separate topic.\n\n"
            "EXAMPLES:\n"
            "Chat History:\n"
            "User: What is virtual memory?\n"
            "Assistant: Virtual memory is a memory management technique...\n"
            "User Question: \"What are its advantages?\"\n"
            '{"query_type": "single_hop", "sub_queries": ["What are the advantages of virtual memory?"]}\n\n'
            "Chat History:\n"
            "User: What is page fault?\n"
            "Assistant: A page fault occurs when a program accesses a page not in RAM...\n"
            "User Question: \"Compare it with thrashing.\"\n"
            '{"query_type": "multi_hop", "sub_queries": ["What is a page fault?", "What is thrashing in operating systems?"]}\n\n'
            f"{history_text}"
            f'User Question: "{query}"\n\n'
            "Respond ONLY with a valid JSON object:"
        )

        try:
            raw_response = self.llm.generate(prompt, context=[])
            cleaned = re.sub(r"```(?:json)?", "", raw_response).strip("` \n")
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                q_type = data.get("query_type", "single_hop")
                sub_q = data.get("sub_queries", [])

                if has_multi_hop_cue and q_type == "single_hop":
                    q_type = "multi_hop"

                if q_type == "multi_hop" and (not isinstance(sub_q, list) or len(sub_q) < 2):
                    sub_q = [
                        f"Information regarding the first aspect of: {query}",
                        f"Information regarding the second aspect of: {query}",
                    ]
                elif q_type == "single_hop":
                    if not isinstance(sub_q, list) or not sub_q or not isinstance(sub_q[0], str):
                        single_res = resolve_reference_fallback(query, chat_history)
                        sub_q = [single_res]
                    else:
                        single_res = resolve_reference_fallback(sub_q[0], chat_history)
                        sub_q = [single_res]

                return QueryAnalysis(query_type=q_type, sub_queries=sub_q)
        except Exception as e:
            print(f"[QueryAnalyzer Warning] Analysis failed, defaulting to single_hop: {e}")

        fallback_q = resolve_reference_fallback(query, chat_history)

        if has_multi_hop_cue:
            return QueryAnalysis(
                query_type="multi_hop",
                sub_queries=[
                    f"Information regarding first topic in: {fallback_q}",
                    f"Information regarding second topic in: {fallback_q}",
                ],
            )

        return QueryAnalysis(query_type="single_hop", sub_queries=[fallback_q])


