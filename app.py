import streamlit as st
import os
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from sql_agent_visual import graph

# --- Page Config ---
st.set_page_config(
    page_title="SQL Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Corporate Chat CSS ---
st.markdown("""
<style>
    /* Global Settings */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 16px;
    }
    
    /* Hide Header */
    header {visibility: hidden;}
    .block-container {padding-top: 1rem;}

    /* Chat Container styling (optional) */
    .stChatMessage {display: none;} /* Hide default if any slip through */
    
    /* CUSTOM MESSAGE BUBBLES */
    .chat-row {
        display: flex;
        margin: 10px 0;
        width: 100%;
    }
    
    .row-reverse {
        flex-direction: row-reverse;
    }
    
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 10px;
        max-width: 70%;
        position: relative;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* User Bubble (Right) */
    .user-bubble {
        background-color: #0078D4; /* Corporate Blue */
        color: white;
        text-align: left; /* Text inside bubble is standard */
        margin-left: auto; /* Push to right */
        border-bottom-right-radius: 2px;
    }
    
    /* AI Bubble (Left) */
    .ai-bubble {
        background-color: #f3f2f1; /* Corporate Light Grey */
        color: #201f1e;
        margin-right: auto;
        border-bottom-left-radius: 2px;
        border: 1px solid #e1dfdd;
    }
    
    /* Tool Output Area */
    .tool-output {
        font-family: 'Consolas', monospace;
        font-size: 0.8rem;
        color: #605e5c;
        background: #faf9f8;
        padding: 8px;
        margin-top: 5px;
        border-left: 3px solid #0078D4;
    }
    
    /* Input Field Styling */
    .stChatInput {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render Chat History (Custom HTML) ---
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f'''<div class="chat-row row-reverse"><div class="chat-bubble user-bubble">{content}</div></div>''', unsafe_allow_html=True)
    elif role == "assistant":
        # Check if content has tool outputs attached (simplified logic)
        # For history, we just show the content. Plot is mostly separate.
        st.markdown(f'''<div class="chat-row"><div class="chat-bubble ai-bubble">{content}</div></div>''', unsafe_allow_html=True)

# Input Area
prompt = st.chat_input("Type your query here...")

if prompt:
    # 1. Append & Display User Message Immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'''<div class="chat-row row-reverse"><div class="chat-bubble user-bubble">{prompt}</div></div>''', unsafe_allow_html=True)

    # 2. Process with Agent (Spinner only, no 'Done' box)
    # We use a placeholder for the streaming AI response to update in real-time
    ai_placeholder = st.empty()
    full_response = ""
    
    # Cleanup old plot
    if os.path.exists("output_plot.png"):
        os.remove("output_plot.png")

    with st.spinner('Processing...'):
        try:
            events = graph.stream(
                {"messages": [("user", prompt)]},
                stream_mode="values"
            )
            
            for event in events:
                msg = event["messages"][-1]
                
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        pass
                    else:
                        full_response = msg.content
                        # Update the placeholder vertically
                        ai_placeholder.markdown(f'''<div class="chat-row"><div class="chat-bubble ai-bubble">{full_response}</div></div>''', unsafe_allow_html=True)
                        
                elif isinstance(msg, ToolMessage):
                    pass # Hide tool outputs

        except Exception as e:
            st.error(f"Error: {e}")

    # 3. Finalize
    # Check for plot
    if os.path.exists("output_plot.png"):
        # Display plot
        st.image("output_plot.png", caption="Analysis Result", use_container_width=True)
        
    # Appending response in history so it shows up next reload
    st.session_state.messages.append({"role": "assistant", "content": full_response})
