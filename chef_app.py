"""
Camera + chat UI for the Personal Chef Agent.

    pip install -U streamlit
    streamlit run chef_app.py

Open the link it prints. On your phone (same Wi-Fi), use the "Network URL".
Note: phone browsers only allow the camera on https or localhost; if the camera
is blocked on your phone, use "Upload a photo" instead.
"""

import base64
import uuid
import streamlit as st
from langgraph.graph import StateGraph, MessagesState, START        # old: no END
from chef_agent import build_graph, ask_chef

st.set_page_config(page_title="Chef Remy", page_icon="🍳")
st.title("🍳 Chef Remy")
st.caption("Snap your fridge, pick a dish, get the recipe.")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
    st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    st.session_state.chat = []
    st.session_state.last_photo = None

with st.expander("📸 Show me your fridge", expanded=not st.session_state.chat):
    tab_cam, tab_upload = st.tabs(["Take a photo", "Upload a photo"])
    with tab_cam:
        shot = st.camera_input("Point at your open fridge")
    with tab_upload:
        upload = st.file_uploader("Fridge photo", type=["jpg", "jpeg", "png", "webp"])

photo = shot or upload
if photo is not None:
    data = photo.getvalue()
    photo_id = hash(data)
    if photo_id != st.session_state.last_photo:  # only scan new photos
        st.session_state.last_photo = photo_id
        st.session_state.chat.append(("user", "📸 *Sent a fridge photo*"))
        with st.spinner("Chef is looking in your fridge..."):
            reply = ask_chef(
                st.session_state.graph, st.session_state.config,
                photo_b64=base64.b64encode(data).decode(),
                mime=photo.type or "image/jpeg",
            )
        st.session_state.chat.append(("assistant", reply))

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

if prompt := st.chat_input("Pick a recipe number, or ask the chef anything"):
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Cooking up an answer..."):
            reply = ask_chef(st.session_state.graph, st.session_state.config, text=prompt)
        st.markdown(reply)
    st.session_state.chat.append(("assistant", reply))