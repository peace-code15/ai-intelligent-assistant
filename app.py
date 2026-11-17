import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from rag_engine import process_uploaded_file, retrieve_context
from tools import AVAILABLE_TOOLS, TOOL_MAP, load_tasks

load_dotenv()

st.set_page_config(page_title="AI Intelligent Assistant", page_icon="🤖", layout="wide")

# Verify required configuration
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Please add it to your .env file.")
    st.stop()

llm_model = os.getenv("LLM_MODEL", "").strip()
if not llm_model:
    st.error("LLM_MODEL not found. Please add it to your .env file.")
    st.stop()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.3)
llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)

def get_message_text(resp) -> str:
    """Safely extracts text whether the response is a string or a list of content blocks."""
    if hasattr(resp, "text") and resp.text:
        return resp.text
    content = resp.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for b in content:
            if isinstance(b, dict) and "text" in b:
                text_parts.append(b["text"])
            elif isinstance(b, str):
                text_parts.append(b)
        return "".join(text_parts) if text_parts else str(content)
    return str(content)

# --- SIDEBAR: Document Intelligence & Live Automation Tracker ---
with st.sidebar:
    st.header("📄 Document Intelligence")
    uploaded_file = st.file_uploader("Upload PDF or TXT for Q&A", type=["pdf", "txt"])
    
    if uploaded_file and uploaded_file.name != st.session_state.doc_name:
        with st.spinner("Indexing document..."):
            try:
                st.session_state.vector_store = process_uploaded_file(uploaded_file)
                st.session_state.doc_name = uploaded_file.name
                st.success(f"Indexed: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.doc_name:
        st.caption(f"Active Knowledge Base: **{st.session_state.doc_name}**")
        if st.button("Clear Document"):
            st.session_state.vector_store = None
            st.session_state.doc_name = None
            st.rerun()

    st.divider()
    st.header("⚡ Automated Action Items")
    st.caption("Ask the assistant in chat to create or complete tasks.")
    tasks = load_tasks()
    if not tasks:
        st.info("No tasks logged yet.")
    else:
        for t in tasks:
            icon = "✅" if t["status"] == "Completed" else "⏳"
            st.markdown(f"**{icon} #{t['id']} {t['title']}**")
            st.caption(f"Priority: {t['priority']} | Due: {t['due_date']}")

    st.divider()
    if st.button("Reset Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN PANEL: AI Chatbot ---
st.title("🤖 AI-Powered Intelligent Assistant")
st.caption("Supports multi-turn conversation, document Q&A (RAG), and automated task workflows.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle User Input
if prompt := st.chat_input("Ask a question, query your document, or say 'Remind me to...'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Retrieve Document Context (if a document is loaded)
            doc_context = retrieve_context(st.session_state.vector_store, prompt)
            
            system_instruction = (
                "You are an intelligent, concise AI assistant. "
                "Use the provided document context if relevant to answer the user's question accurately. "
                "If the user asks to create, log, schedule, or complete a task/reminder, use your available tools."
            )
            if doc_context:
                system_instruction += f"\n\n### Retrieved Document Context:\n{doc_context}"

            # 2. Build Conversation History
            lc_messages = [SystemMessage(content=system_instruction)]
            for m in st.session_state.messages:
                if m["role"] == "user":
                    lc_messages.append(HumanMessage(content=m["content"]))
                else:
                    lc_messages.append(AIMessage(content=m["content"]))

            # 3. Invoke LLM & Execute Any Triggered Automation Tools
            response = llm_with_tools.invoke(lc_messages)
            
            if response.tool_calls:
                lc_messages.append(response)
                for tool_call in response.tool_calls:
                    selected_tool = TOOL_MAP[tool_call["name"]]
                    tool_output = selected_tool.invoke(tool_call["args"])
                    lc_messages.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )
                final_response = llm_with_tools.invoke(lc_messages)
                reply_text = get_message_text(final_response)
            else:
                reply_text = get_message_text(response)

            st.markdown(reply_text)
            st.session_state.messages.append({"role": "assistant", "content": reply_text})
            
            # Rerun if a tool ran so the sidebar task list updates immediately
            if response.tool_calls:
                st.rerun()