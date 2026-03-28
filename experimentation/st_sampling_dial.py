import asyncio
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from wikipedia_tool.main import WikipediaToolKit

# --- 1. UI Setup ---
st.set_page_config(page_title="ReAct Agent Sandbox", layout="wide")
st.title("🔌 ReAct Agent Sandbox")

# --- 2. The Control Room (Sidebar) ---
with st.sidebar:
    st.header("Server Connection")
    api_base = st.text_input("API Base URL", value="http://localhost:8090/v1")
    
    st.header("Model Parameters")
    # These variables update dynamically whenever you move the slider
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=1.0)
    max_tokens = st.number_input("Max Tokens", value=512)
    top_p = st.slider("Top-p (Nucleus Sampling)", min_value=0.0, max_value=1.0, value=0.95)
    top_k = st.slider("Top-k Sampling", min_value=0, max_value=1000, value=40)
    min_p = st.slider("Min-p", min_value=0.0, max_value=1.0, value=0.01)
    
    st.header("System Prompt")
    system_instructions = st.text_area(
        "System Instructions",
        value="You are a helpful assistant. Answer the user's questions as best you can.",
        height=150
    )

# --- 3. Chat Memory (Session State) ---
# Because Streamlit reruns top-to-bottom on every click, 
# we use session_state to act as a permanent notebook for the chat history.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw all previous messages on every script rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def get_last_user_message() -> str | None:
    """Return the most recent user prompt from chat history."""
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "user":
            return msg.get("content")
    return None

# --- 4. Execution (Chat Input) ---
async def main():
    last_user_prompt = get_last_user_message()
    reroll_clicked = st.button(
        "Reroll last response",
        disabled=last_user_prompt is None,
        help="Regenerate the assistant answer using current sampling settings.",
    )

    # This block triggers when the user hits 'Enter' in the chat box
    prompt = st.chat_input("Test your agent here...")

    if prompt or reroll_clicked:
        is_reroll = bool(reroll_clicked and not prompt)
        question = prompt if prompt else last_user_prompt

        if question is None:
            return
        
        # Display user input immediately and save to memory
        if not is_reroll:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

        # Setup the assistant's response area
        with st.chat_message("assistant"):
            with st.spinner("Agent is running..."):
                sampling_params = {
                    "top_k": top_k,
                    "min_p": min_p
                }

                llm = ChatOpenAI(
                    model="unsloth/MiniMax-M2.1-GGUF",   
                    api_key="fake_key_for_llama_cpp",
                    base_url=api_base,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    extra_body=sampling_params
                )

                USER_AGENT = "TriviaQA (dzureca7@gmail.com)"

                wikipedia = WikipediaToolKit(USER_AGENT)
                tools = [wikipedia.search, wikipedia.inspect]
                agent = create_agent(llm, tools)

                def get_final_answer(result: dict) -> str:
                    """Parse final answer from ReAct agent"""
                    messages = result.get("messages", [])
                    # Search backwards for the last AI message with actual text content
                    for msg in reversed(messages):
                        # In LangChain, messages are objects, so we access .type and .content
                        if msg.type == "ai" and msg.content:
                            return msg.content
                    return "No answer found."

                async def run_agent(question: str, system_instructions: str):
                    """Run the ReAct agent"""
                    result = await agent.ainvoke({"messages": [
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": question}
                        ]})
                    # Optional: print raw result for debugging
                    # print(result)
                    
                    final_answer = get_final_answer(result)
                    
                    return final_answer
                
                # Dummy response so the UI runs out of the box
                final_answer = await run_agent(question, system_instructions)
                # ==========================================
                
                # Display and save the final output
                st.markdown(final_answer)
                if is_reroll and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                    st.session_state.messages[-1]["content"] = final_answer
                else:
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

if __name__ == "__main__":
    asyncio.run(main())
