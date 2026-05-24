from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from bwa_backend import app

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


def list_past_blogs() -> List[Path]:
    files = [p for p in Path(".").glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def bundle_zip(md_text: str, md_filename: str) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        images_dir = Path("images")
        if images_dir.exists():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


# ─────────────────────────────────────────
# Markdown renderer with local image support
# ─────────────────────────────────────────
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def render_markdown_with_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        if md[last:m.start()]:
            parts.append(("md", md[last:m.start()]))
        parts.append(("img", f"{m.group('alt')}|||{m.group('src')}"))
        last = m.end()
    if md[last:]:
        parts.append(("md", md[last:]))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload)
            i += 1
            continue

        alt, src = payload.split("|||", 1)
        caption = None

        # Check next part for italic caption line
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            first_line = nxt.splitlines()[0].strip() if nxt.strip() else ""
            m = _CAPTION_RE.match(first_line)
            if m:
                caption = m.group("cap").strip()
                rest = "\n".join(nxt.splitlines()[1:])
                parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or alt or None, use_container_width=True)
        else:
            img_path = Path(src.lstrip("./")).resolve()
            if img_path.exists():
                st.image(str(img_path), caption=caption or alt or None, use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}`")
        i += 1


# ─────────────────────────────────────────
# Graph streaming helper
# ─────────────────────────────────────────
def run_graph(inputs: Dict[str, Any]):
    """
    Invoke the graph and yield (node_name, partial_state) as nodes complete.
    Falls back to plain invoke if streaming is not available.
    """
    try:
        current: Dict[str, Any] = {}
        for chunk in app.stream(inputs, stream_mode="updates"):
            if isinstance(chunk, dict):
                node_name = next(iter(chunk.keys()), "unknown")
                node_out = next(iter(chunk.values()), {})
                if isinstance(node_out, dict):
                    current.update(node_out)
                yield node_name, current
    except Exception:
        out = app.invoke(inputs)
        yield "done", out


# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
)

st.title("✍️ Blog Writing Agent")
st.caption("Powered by LangGraph · GitHub Models · HuggingFace · Tavily")

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Generate New Blog")

    topic = st.text_area(
        "Topic",
        placeholder="e.g. State of Agentic AI Frameworks in 2025",
        height=100,
    )

    as_of = st.date_input(
        "As-of date",
        value=date.today(),
        help="Used for research recency filtering."
    )

    generate_images = st.toggle(
        "Generate Images",
        value=True,
        help="Use HuggingFace FLUX to generate diagrams. Requires HF_TOKEN."
    )

    run_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)

    st.divider()

    # ── Past blogs ──
    st.subheader("📂 Past Blogs")
    past_files = list_past_blogs()

    if not past_files:
        st.caption("No saved blogs found (*.md in project folder).")
    else:
        options: Dict[str, Path] = {}
        for p in past_files[:30]:
            try:
                md_text = p.read_text(encoding="utf-8", errors="replace")
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title[:40]}  ·  {p.name}"
            options[label] = p

        selected_label = st.radio(
            "Select a blog",
            list(options.keys()),
            index=0,
            label_visibility="collapsed",
        )

        if st.button("📖 Load Selected", use_container_width=True):
            p = options[selected_label]
            md_text = p.read_text(encoding="utf-8", errors="replace")
            st.session_state["last_out"] = {
                "plan": None,
                "evidence": [],
                "image_specs": [],
                "mode": "loaded",
                "needs_research": False,
                "final": md_text,
            }
            st.rerun()

# ─────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "run_logs" not in st.session_state:
    st.session_state["run_logs"] = []

# ─────────────────────────────────────────
# Run graph on button click
# ─────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    st.session_state["run_logs"] = []
    st.session_state["last_out"] = None

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        # Skip image generation if toggle is off
        "image_specs": [] if not generate_images else [],
        "final": "",
    }

    # If images are disabled, patch the decide_images node to return empty specs
    # We do this by passing a sentinel in image_specs and checking in the backend
    # Simpler: just override generate_and_place_images at runtime via a flag
    if not generate_images:
        os.environ["BWA_SKIP_IMAGES"] = "1"
    else:
        os.environ.pop("BWA_SKIP_IMAGES", None)

    status_box = st.status("🔄 Running graph...", expanded=True)
    progress_placeholder = st.empty()
    current_state: Dict[str, Any] = {}

    with status_box:
        for node_name, partial_state in run_graph(inputs):
            current_state.update(partial_state)
            st.write(f"✅ Node completed: `{node_name}`")

            # Live summary
            summary = {}
            if current_state.get("mode"):
                summary["mode"] = current_state["mode"]
            if current_state.get("needs_research") is not None:
                summary["needs_research"] = current_state["needs_research"]
            ev = current_state.get("evidence") or []
            if ev:
                summary["evidence"] = len(ev)
            plan = current_state.get("plan")
            if plan:
                pdict = plan.model_dump() if hasattr(plan, "model_dump") else plan
                summary["sections"] = len(pdict.get("tasks", []))
                summary["blog_kind"] = pdict.get("blog_kind")
            specs = current_state.get("image_specs") or []
            if specs:
                summary["images_planned"] = len(specs)
            if summary:
                progress_placeholder.json(summary)

            st.session_state["run_logs"].append(
                f"[{node_name}] {json.dumps(partial_state, default=str)[:400]}"
            )

    # Extract final output
    final_out = current_state if current_state.get("final") else app.invoke(inputs)
    st.session_state["last_out"] = final_out
    status_box.update(label="✅ Blog generated!", state="complete", expanded=False)
    st.rerun()

# ─────────────────────────────────────────
# Display results
# ─────────────────────────────────────────
out = st.session_state.get("last_out")

if out:
    plan_obj = out.get("plan")
    final_md = out.get("final") or ""
    mode = out.get("mode", "")
    evidence = out.get("evidence") or []
    image_specs = out.get("image_specs") or []

    # Resolve blog title
    if plan_obj:
        if hasattr(plan_obj, "blog_title"):
            blog_title = plan_obj.blog_title
        elif isinstance(plan_obj, dict):
            blog_title = plan_obj.get("blog_title", "blog")
        else:
            blog_title = extract_title_from_md(final_md, "blog")
    else:
        blog_title = extract_title_from_md(final_md, "blog")

    md_filename = f"{safe_slug(blog_title)}.md"

    # ── Tabs ──
    tab_preview, tab_plan, tab_evidence, tab_images, tab_logs = st.tabs([
        "📝 Preview",
        "🧩 Plan",
        "🔎 Evidence",
        "🖼️ Images",
        "🧾 Logs",
    ])

    # ── Preview Tab ──
    with tab_preview:
        if not final_md:
            st.info("No blog generated yet.")
        else:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col2:
                bundle = bundle_zip(final_md, md_filename)
                st.download_button(
                    "📦 Download Bundle (MD + Images)",
                    data=bundle,
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            with col3:
                st.metric("Characters", f"{len(final_md):,}")

            st.divider()
            render_markdown_with_images(final_md)

    # ── Plan Tab ──
    with tab_plan:
        if not plan_obj:
            st.info("No plan available.")
        else:
            pdict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj
            if isinstance(pdict, dict):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Title:** {pdict.get('blog_title', '')}")
                    st.markdown(f"**Audience:** {pdict.get('audience', '')}")
                with col2:
                    st.markdown(f"**Tone:** {pdict.get('tone', '')}")
                    st.markdown(f"**Blog kind:** `{pdict.get('blog_kind', '')}`")
                    st.markdown(f"**Mode:** `{mode}`")

                tasks = pdict.get("tasks", [])
                if tasks:
                    st.subheader(f"{len(tasks)} Sections")
                    df = pd.DataFrame([
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "words": t.get("target_words"),
                            "code": "✓" if t.get("requires_code") else "",
                            "research": "✓" if t.get("requires_research") else "",
                            "citations": "✓" if t.get("requires_citations") else "",
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    with st.expander("Full task details (JSON)"):
                        st.json(tasks)

    # ── Evidence Tab ──
    with tab_evidence:
        if not evidence:
            st.info(
                "No evidence collected. "
                "This is expected for closed_book topics or when TAVILY_API_KEY is not set."
            )
        else:
            st.metric("Evidence items", len(evidence))
            rows = []
            for e in evidence:
                edict = e.model_dump() if hasattr(e, "model_dump") else e
                rows.append({
                    "title": edict.get("title", "")[:60],
                    "published_at": edict.get("published_at") or "unknown",
                    "source": edict.get("source") or "",
                    "url": edict.get("url", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Images Tab ──
    with tab_images:
        images_dir = Path("images")
        if not image_specs and not images_dir.exists():
            st.info("No images generated for this blog.")
        else:
            if image_specs:
                st.subheader("Image Plan")
                for spec in image_specs:
                    sdict = spec if isinstance(spec, dict) else spec
                    with st.expander(f"{sdict.get('placeholder')} → {sdict.get('filename')}"):
                        st.markdown(f"**Alt:** {sdict.get('alt')}")
                        st.markdown(f"**Caption:** {sdict.get('caption')}")
                        st.markdown(f"**Prompt:** {sdict.get('prompt')}")

            if images_dir.exists():
                img_files = sorted([p for p in images_dir.iterdir() if p.is_file()])
                if img_files:
                    st.subheader(f"{len(img_files)} Generated Image(s)")
                    cols = st.columns(min(len(img_files), 3))
                    for i, p in enumerate(img_files):
                        with cols[i % 3]:
                            st.image(str(p), caption=p.name, use_container_width=True)

                    # Download images zip
                    buf = BytesIO()
                    with zipfile.ZipFile(buf, "w") as z:
                        for p in img_files:
                            z.write(p, arcname=str(p))
                    st.download_button(
                        "⬇️ Download All Images",
                        data=buf.getvalue(),
                        file_name="images.zip",
                        mime="application/zip",
                    )

    # ── Logs Tab ──
    with tab_logs:
        logs = st.session_state.get("run_logs", [])
        if not logs:
            st.info("No logs from current session. Logs appear after generating a blog.")
        else:
            st.text_area(
                "Node execution log",
                value="\n\n".join(logs[-60:]),
                height=500,
            )
            if st.button("Clear logs"):
                st.session_state["run_logs"] = []
                st.rerun()

else:
    # Landing state — no blog generated yet
    st.info("👈 Enter a topic in the sidebar and click **Generate Blog** to get started.")

    st.markdown("""
    ### How it works

    1. **Router** classifies your topic as `closed_book`, `hybrid`, or `open_book`
    2. **Research** (optional) searches the web via Tavily for fresh evidence
    3. **Orchestrator** creates a structured blog plan with 5-7 sections
    4. **Workers** write each section in parallel with rate-limit handling
    5. **Image AI** decides where diagrams help and generates them via HuggingFace FLUX
    6. **Reducer** assembles everything into a final Markdown file

    ### Requirements
    | Key | Where to get it |
    |---|---|
    | `GITHUB_TOKEN` | github.com/marketplace/models |
    | `TAVILY_API_KEY` | app.tavily.com (optional, for research) |
    | `HF_TOKEN` | huggingface.co/settings/tokens (optional, for images) |
    """)