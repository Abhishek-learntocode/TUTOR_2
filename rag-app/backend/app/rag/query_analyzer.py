import json
import re
from app.models.state import QueryAnalysis


class QueryAnalyzer:
    """Analyzes queries using LLM + rule heuristics to classify as single_hop or multi_hop with sub-queries."""

    def __init__(self, llm):
        self.llm = llm

    def analyze(self, query: str) -> QueryAnalysis:
        query_lower = query.lower()
        has_multi_hop_cue = (
            "compare" in query_lower
            or "difference between" in query_lower
            or " versus " in query_lower
            or " vs " in query_lower
            or bool(re.search(r"chapter\s+\d+.*chapter\s+\d+", query, re.I))
        )

        prompt = (
            "You are a Query Analyzer for a RAG system.\n"
            "Classify the user question as 'single_hop' or 'multi_hop' and generate sub_queries.\n\n"
            "RULES:\n"
            "1. Default to 'single_hop' for questions asking about a single concept, definition, or direct topic.\n"
            "2. Classify as 'multi_hop' ONLY when answering requires combining or comparing separate topics/chapters.\n"
            "3. For single_hop, sub_queries must contain ONLY the original question.\n"
            "4. For multi_hop, generate 2-3 focused sub-queries for each separate topic.\n\n"
            "EXAMPLES:\n"
            'Question: "What is virtual memory?"\n'
            '{"query_type": "single_hop", "sub_queries": ["What is virtual memory?"]}\n\n'
            'Question: "Compare CPU scheduling in Chapter 4 with memory management in Chapter 7."\n'
            '{"query_type": "multi_hop", "sub_queries": ["What is CPU scheduling in Chapter 4?", "What is memory management in Chapter 7?"]}\n\n'
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
                    sub_q = [query]

                return QueryAnalysis(query_type=q_type, sub_queries=sub_q)
        except Exception as e:
            print(f"[QueryAnalyzer Warning] Analysis failed, defaulting to single_hop: {e}")

        if has_multi_hop_cue:
            return QueryAnalysis(
                query_type="multi_hop",
                sub_queries=[
                    f"Information regarding first topic in: {query}",
                    f"Information regarding second topic in: {query}",
                ],
            )

        return QueryAnalysis(query_type="single_hop", sub_queries=[query])
