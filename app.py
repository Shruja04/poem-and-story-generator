import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="TextWeaver: AI Driven Poem and Story Generator",
    page_icon="✨",
    layout="centered"
)

# Custom UI Styling
st.markdown("""
<style>

.main {
    background-color: #f4f6fb;
}

h1 {
    color: #2E4053;
    text-align: center;
}

.stButton>button {
    background-color: #5A67D8;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #434190;
    color: white;
}

.stSidebar {
    background-color: #eef1f7;
}

div[data-testid="stSuccessMessage"] {
    background-color: #E6FFFA;
    border-left: 6px solid #38B2AC;
}

</style>
""", unsafe_allow_html=True)


# Title
st.title("✨ TextWeaver: AI Driven Poem and Story Generator")

st.markdown(
    "<center>Generate creative poems and short stories using a fine-tuned GPT-2 Transformer model.</center>",
    unsafe_allow_html=True
)


# Load model
@st.cache_resource
def load_model():
    generator = pipeline(
        "text-generation",
        model="poem_story_model",
        tokenizer="poem_story_model"
    )
    return generator


generator = load_model()


# Sidebar settings
st.sidebar.header("⚙️ Generation Settings")

mode = st.sidebar.radio(
    "Choose generation type:",
    ["Story", "Poem"]
)

temperature = st.sidebar.slider(
    "Creativity Level",
    0.5,
    1.2,
    0.9
)

length = st.sidebar.slider(
    "Output Length",
    50,
    150,
    100
)


# Prompt input
prompt = st.text_input("✍️ Enter your prompt:")


# Generate button
if st.button("🚀 Generate Text"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt first!")

    else:

        formatted_prompt = f"{mode}: {prompt}"

        output = generator(
            formatted_prompt,
            max_length=length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2
        )

        st.subheader("📜 Generated Output")
        st.success(output[0]["generated_text"])


# Footer
st.markdown("---")

st.caption("Built using GPT-2 + HuggingFace Transformers + Streamlit")