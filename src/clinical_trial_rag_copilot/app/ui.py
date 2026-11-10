import os
import streamlit as st
import pandas as pd

from clinical_trial_rag_copilot.agent.copilot import ClinicalGraphRAGCopilot
from clinical_trial_rag_copilot.graph.connection import get_db

st.set_page_config(
    page_title="Cardiology Clinical Trial & Pharmacology GraphRAG",
    page_icon="🫀",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 18px;
        border-left: 4px solid #0d6efd;
        margin-bottom: 10px;
    }
    .badge-passed {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-warning {
        background-color: #fff3cd;
        color: #664d03;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_copilot():
    """Cache the GraphRAG copilot instance."""
    return ClinicalGraphRAGCopilot()


@st.cache_data(ttl=300)
def get_graph_stats():
    """Fetch live counts of nodes and relationships from Neo4j."""
    db = get_db()
    try:
        with db.driver.session() as session:
            node_res = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"
            )
            node_counts = {r["label"]: r["count"] for r in node_res}

            rel_res = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS count ORDER BY count DESC"
            )
            rel_counts = {r["rel"]: r["count"] for r in rel_res}
            return node_counts, rel_counts
    except Exception as e:
        return {}, {}


# Header
st.title("🫀 Cardiology Clinical Trial & Pharmacology GraphRAG Copilot")
st.caption("Dual-Indexed (Neo4j AuraDB + Pinecone Vector) Multi-Hop Assistant | Post-Cutoff >= June 2024")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    provider = st.selectbox(
        "Embedding & LLM Provider",
        ["Local & Free (Pure-Python)", "Google Gemini (Free Tier)", "OpenAI (Cloud)"],
        index=0,
    )
    if provider == "Google Gemini (Free Tier)":
        os.environ["EMBEDDING_PROVIDER"] = "google"
        gemini_key = st.text_input("GOOGLE_API_KEY", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key
    elif provider == "OpenAI (Cloud)":
        os.environ["EMBEDDING_PROVIDER"] = "openai"
        openai_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
    else:
        os.environ["EMBEDDING_PROVIDER"] = "local"

    top_k = st.slider("Retrieval Depth (Top-K)", min_value=3, max_value=12, value=6)

    st.divider()
    st.subheader("📊 Live Knowledge Graph Status")
    node_counts, rel_counts = get_graph_stats()
    if node_counts:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Trials", f"{node_counts.get('Trial', 0):,}")
            st.metric("Study Arms", f"{node_counts.get('ArmGroup', 0):,}")
            st.metric("Drugs", f"{node_counts.get('Drug', 0):,}")
        with c2:
            st.metric("Target Genes", f"{node_counts.get('Gene', 0):,}")
            st.metric("PubMed Papers", f"{node_counts.get('Paper', 0):,}")
            st.metric("Relationships", f"{sum(rel_counts.values()):,}")
    else:
        st.warning("Neo4j database status unavailable.")

# Main Interface
col_samples, col_input = st.columns([1, 2])

with col_samples:
    st.subheader("💡 Sample Clinical Inquiries")
    sample_queries = [
        "Which clinical trials evaluate Amiodarone targeting KCNH2 in patients with Atrial Fibrillation?",
        "What study arms test Vericiguat and what are the primary endpoints and age criteria?",
        "What research papers report QT Prolongation or Torsades de Pointes for antiarrhythmic drugs?",
        "Find heart failure trials for elderly patients (aged >= 65) testing mineralocorticoid receptor blockers.",
    ]
    selected_sample = st.radio("Select a pre-configured clinical scenario:", sample_queries)

with col_input:
    st.subheader("🔎 Ask Question")
    user_query = st.text_area("Enter your clinical or pharmacological inquiry:", value=selected_sample, height=100)
    ask_button = st.button("🚀 Analyze & Synthesize Multi-Hop Answer", type="primary")

if ask_button and user_query.strip():
    copilot = get_copilot()
    with st.spinner("Executing Dual-Indexed Graph + Vector Retrieval & Factuality Guardrail..."):
        ans = copilot.ask(user_query, top_k=top_k)

    # Output Presentation
    st.divider()
    
    # Guardrail Status Pill
    if ans.guardrail_status == "PASSED":
        st.markdown(
            f"### Guardrail Audit: <span class='badge-passed'>✔ KNOWLEDGE GRAPH VERIFIED ({len(ans.verified_trials)} Trials Confirmed)</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"### Guardrail Audit: <span class='badge-warning'>⚠ GUARDRAIL WARNING ({ans.guardrail_status})</span>",
            unsafe_allow_html=True,
        )

    tab_ans, tab_guardrail, tab_graph, tab_chunks = st.tabs([
        "📝 Synthesized Medical Response",
        "🛡️ Self-RAG Guardrail Details",
        "🕸️ Graph Traversal Paths",
        "📄 Vector Text Chunks",
    ])

    with tab_ans:
        st.markdown(ans.answer)

    with tab_guardrail:
        st.subheader("Knowledge Graph Factuality Verification Audit")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("#### Confirmed Clinical Trial Identifiers")
            if ans.verified_trials:
                for t in ans.verified_trials:
                    st.success(f"✔ **{t}**: Verified in Neo4j AuraDB (`(:Trial {{nct_id: '{t}'}})` )")
            else:
                st.info("No specific NCT IDs cited in answer.")
        with col_g2:
            st.markdown("#### Confirmed Pharmacology Interactions")
            if ans.verified_interactions:
                for i in ans.verified_interactions:
                    st.success(f"✔ **{i['drug']}** $\\rightarrow$ **{i['gene']}** (Edge: `[:INHIBITS {{action: '{i['action']}'}}]`)")
            else:
                st.info("No specific drug-gene edges cited.")

    with tab_graph:
        st.subheader("Retrieved Neo4j Subgraph Paths")
        st.write("Cross-database links discovered via multi-hop Cypher queries:")
        st.json({
            "retrieved_ncts": ans.retrieved_ncts,
            "retrieved_pmids": ans.retrieved_pmids,
            "verified_interactions": ans.verified_interactions,
        })

    with tab_chunks:
        st.subheader("Ranked Vector Document Chunks (RRF)")
        st.write(f"Top {len(ans.retrieved_ncts)} fused NCT records and PMIDs prioritized for LLM context assembly.")
        st.table(pd.DataFrame({
            "Rank": list(range(1, len(ans.retrieved_ncts) + 1)),
            "Document Identifier": ans.retrieved_ncts,
        }))
