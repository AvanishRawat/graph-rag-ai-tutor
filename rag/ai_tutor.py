from typing import Dict, Any
from project.rag import retriever, prompt as prompt_mod, generator
import networkx as nx

class AITutor:
    def __init__(self, top_k_concepts: int = 3, prereq_depth: int = 2):
        self.top_k = top_k_concepts
        self.prereq_depth = prereq_depth

    def answer(self, question: str) -> Dict[str, Any]:
        out = retriever.retrieve_subgraph_for_query(question, top_k=self.top_k, prereq_depth=self.prereq_depth)
        sub = out["subgraph"]
        seeds = out["seed_concepts"]

        prompt_text = prompt_mod.build_generation_prompt(question, sub, seeds)

        answer_text = generator.ask_ollama(prompt_text)

        used_concepts = [(c, sub.nodes[c].get("title","")) for c in out["used_concepts"]]
        used_resources = [(r, sub.nodes[r].get("title",""), sub.nodes[r].get("url","")) for r in out["used_resources"]]
        used_examples = [(e, sub.nodes[e].get("text","")[:300]) for e in out["used_examples"]]

        result = {
            "question": question,
            "system_prompt": prompt_mod.SYSTEM_PROMPT.strip(),
            "seed_concepts": seeds,
            "ranked": out.get("ranked", []),
            "subgraph_nodes": list(sub.nodes()),
            "subgraph_edges": [(u,v,d) for u,v,d in sub.edges(data=True)],
            "answer": answer_text,
            "used_concepts": used_concepts,
            "used_resources": used_resources,
            "used_examples": used_examples,
            "prompt_text": prompt_text[:8000],
        }
        return result
