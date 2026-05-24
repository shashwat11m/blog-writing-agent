from __future__ import annotations

import operator
import os
import re
import io
import time
import random
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# load_dotenv() works locally from .env file
# On Streamlit Cloud the keys are already in the environment
load_dotenv()

# ─────────────────────────────────────────
# 1) LLM
# ─────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference"
)

# ─────────────────────────────────────────
# 2) Schemas
# ─────────────────────────────────────────
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should be able to do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target word count (120–550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str
    filename: str
    alt: str
    caption: str
    prompt: str
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1536x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


# ─────────────────────────────────────────
# 3) State
# ─────────────────────────────────────────
class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str


# ─────────────────────────────────────────
# 4) Retry helper
# ─────────────────────────────────────────
def call_llm_with_retry(messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "too many" in str(e).lower():
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit. Waiting {wait:.1f}s (retry {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded")


# ─────────────────────────────────────────
# 5) Router
# ─────────────────────────────────────────
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/latest/pricing/policy.

If needs_research=true output 3-5 high-signal, specific queries.
Output must match the RouterDecision schema.
"""


def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
    ])
    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# ─────────────────────────────────────────
# 6) Research
# ─────────────────────────────────────────
def _tavily_search(query: str, max_results: int = 4) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_tavily import TavilySearch
        tool = TavilySearch(max_results=max_results)
        results = tool.invoke({"query": query})
        raw_list = results if isinstance(results, list) else results.get("results", [])
        out = []
        for r in raw_list or []:
            snippet = r.get("content") or r.get("snippet") or ""
            out.append({
                "title": (r.get("title") or "")[:120],
                "url": r.get("url") or "",
                "snippet": snippet[:300],
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            })
        return out
    except Exception as e:
        print(f"Tavily search failed: {e}")
        return []


def _iso_to_date(s: Optional[str]):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _truncate_raw_results(raw_results: List[dict], max_items: int = 15) -> List[dict]:
    return [
        {
            "title": (r.get("title") or "")[:100],
            "url": (r.get("url") or "")[:200],
            "snippet": (r.get("snippet") or "")[:250],
            "published_at": r.get("published_at"),
        }
        for r in raw_results[:max_items]
    ]


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.
Given raw web search results, produce a deduplicated list of EvidenceItem objects.
Rules:
- Only include items with a non-empty url.
- Prefer authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD only if reliably inferable, else null.
- Keep snippets short.
- Deduplicate by URL.
- Return at most 12 items.
Output must match the EvidencePack schema.
"""


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:5]
    raw_results: List[dict] = []
    for i, q in enumerate(queries):
        print(f"Searching ({i+1}/{len(queries)}): {q}")
        raw_results.extend(_tavily_search(q, max_results=4))
        time.sleep(1)
    if not raw_results:
        return {"evidence": []}
    trimmed = _truncate_raw_results(raw_results, max_items=15)
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=(
            f"As-of date: {state['as_of']}\n"
            f"Recency days: {state['recency_days']}\n\n"
            f"Raw results (trimmed):\n{trimmed}"
        )),
    ])
    dedup: dict = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())
    if state.get("mode") == "open_book":
        as_of_date = date.fromisoformat(state["as_of"])
        cutoff = as_of_date - timedelta(days=int(state["recency_days"]))
        evidence = [
            e for e in evidence
            if (d := _iso_to_date(e.published_at)) and d >= cutoff
        ]
    print(f"Evidence collected: {len(evidence)} items")
    return {"evidence": evidence}


# ─────────────────────────────────────────
# 7) Orchestrator
# ─────────────────────────────────────────
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5-7 sections, each with goal + 3-6 bullets + target_words.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Plan must include at least 2 of: MWE/code, edge cases, perf/cost, security, debugging.

Grounding:
- closed_book: evergreen only.
- hybrid: use evidence for fresh examples, mark requires_citations=True.
- open_book: blog_kind=news_roundup, summarize events + implications, no tutorials.

Output must match Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None
    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {mode}\n"
            f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
            f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
            f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
        )),
    ])
    if forced_kind:
        plan.blog_kind = "news_roundup"
    print(f"Plan: '{plan.blog_title}' — {len(plan.tasks)} sections")
    return {"plan": plan}


# ─────────────────────────────────────────
# 8) Fanout
# ─────────────────────────────────────────
def fanout(state: State):
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "as_of": state["as_of"],
            "recency_days": state["recency_days"],
            "plan": state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
            "delay": i * 4,
        })
        for i, task in enumerate(state["plan"].tasks)
    ]


# ─────────────────────────────────────────
# 9) Worker
# ─────────────────────────────────────────
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order, stay within target words ±15%.
- Output only section markdown starting with ## <Section Title>.
- news_roundup: summarize events and implications, no tutorials.
- open_book: only cite URLs from provided Evidence.
- requires_citations=true: cite evidence URLs as ([Source](URL)).
- requires_code=true: include at least one minimal code snippet.

Style: short paragraphs, bullets where helpful, code fences for code.
Avoid fluff and marketing language.
"""


def worker_node(state: dict) -> dict:
    delay = state.get("delay", 0)
    if delay > 0:
        time.sleep(delay)
    task = Task(**state["task"])
    plan = Plan(**state["plan"])
    evidence = [EvidenceItem(**e) for e in state.get("evidence", [])]
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    ) if evidence else "None provided."
    messages = [
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog title: {plan.blog_title}\n"
            f"Audience: {plan.audience}\n"
            f"Tone: {plan.tone}\n"
            f"Blog kind: {plan.blog_kind}\n"
            f"Topic: {state['topic']}\n"
            f"Mode: {state.get('mode')}\n"
            f"As-of: {state.get('as_of')}\n\n"
            f"Section title: {task.title}\n"
            f"Goal: {task.goal}\n"
            f"Target words: {task.target_words}\n"
            f"requires_research: {task.requires_research}\n"
            f"requires_citations: {task.requires_citations}\n"
            f"requires_code: {task.requires_code}\n"
            f"Bullets:{bullets_text}\n\n"
            f"Evidence:\n{evidence_text}\n"
        )),
    ]
    response = call_llm_with_retry(messages)
    return {"sections": [(task.id, str(response.content).strip())]}


# ─────────────────────────────────────────
# 10) Merge content
# ─────────────────────────────────────────
def merge_content(state: State) -> dict:
    plan = state["plan"]
    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered).strip()
    return {"merged_md": f"# {plan.blog_title}\n\n{body}\n"}


# ─────────────────────────────────────────
# 11) Decide images
# ─────────────────────────────────────────
DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide whether diagrams or images would materially improve understanding.

Rules:
- Add at most 3 images. Fewer is fine.
- Only add where they genuinely clarify: flow diagrams, architecture, comparisons.
- Do NOT add decorative images.
- Insert placeholders exactly as [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- Write detailed, specific prompts for technical schematic diagrams.
- If no images needed, return original markdown unchanged and images=[].

Output must match GlobalImagePlan schema.
"""


def decide_images(state: State) -> dict:
    plan = state["plan"]
    planner = llm.with_structured_output(GlobalImagePlan)
    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Blog kind: {plan.blog_kind}\n"
            f"Topic: {state['topic']}\n\n"
            f"{state['merged_md']}"
        )),
    ])
    print(f"Image plan: {len(image_plan.images)} image(s)")
    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


# ─────────────────────────────────────────
# 12) Image generation
# ─────────────────────────────────────────
def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def _generate_image_bytes(prompt: str, max_retries: int = 3) -> bytes:
    from huggingface_hub import InferenceClient
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set in environment.")
    client = InferenceClient(provider="hf-inference", api_key=hf_token)
    model = "black-forest-labs/FLUX.1-schnell"
    for attempt in range(max_retries):
        try:
            image = client.text_to_image(prompt=prompt, model=model)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            err_str = str(e)
            if "loading" in err_str.lower() or "503" in err_str:
                if attempt < max_retries - 1:
                    time.sleep(30 + random.uniform(1, 5))
                    continue
            if "429" in err_str:
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) + random.uniform(1, 3))
                    continue
            raise RuntimeError(f"Image generation failed: {err_str}") from e
    raise RuntimeError("Image generation failed after all retries.")


def _is_cloud() -> bool:
    """Returns True when running on Streamlit Cloud."""
    return os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs") or []

    if os.environ.get("BWA_SKIP_IMAGES") == "1" or not image_specs:
        md = re.sub(r"\[\[IMAGE_\d+\]\]", "", md)
        md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
        # Only write to disk locally
        if not _is_cloud():
            filename = f"{_safe_slug(plan.blog_title)}.md"
            Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename
        print(f"Generating image: {placeholder} → images/{filename}")
        if out_path.exists():
            print(f"  Reusing existing images/{filename}")
            img_md = f"\n![{spec['alt']}](images/{filename})\n*{spec['caption']}*\n"
            md = md.replace(placeholder, img_md)
            continue
        try:
            img_bytes = _generate_image_bytes(spec["prompt"])
            out_path.write_bytes(img_bytes)
            print(f"  Saved ({len(img_bytes):,} bytes)")
            img_md = f"\n![{spec['alt']}](images/{filename})\n*{spec['caption']}*\n"
            md = md.replace(placeholder, img_md)
            time.sleep(3)
        except Exception as e:
            print(f"  Failed: {e}")
            fallback = (
                f"\n> **[Image not generated]** *{spec.get('caption', '')}*\n"
                f"> **Description:** {spec.get('alt', '')}\n"
                f"> **Error:** {str(e)[:120]}\n"
            )
            md = md.replace(placeholder, fallback)

    md = re.sub(r"\[\[IMAGE_\d+\]\]", "", md)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"

    if not _is_cloud():
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")

    return {"final": md}


# ─────────────────────────────────────────
# 13) Build reducer subgraph
# ─────────────────────────────────────────
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


# ─────────────────────────────────────────
# 14) Build main graph
# ─────────────────────────────────────────
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges(
    "router", route_next,
    {"research": "research", "orchestrator": "orchestrator"}
)
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()