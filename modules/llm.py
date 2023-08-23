import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
plugin_info = {"name": "StableLM"}

# Load model and tokenizer
tokenizer = None
model = None
def load_models():
    global model, tokenizer
    if model == None:
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-7b-v2")
        model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-base-alpha-7b-v2", trust_remote_code=True, torch_dtype="auto")
        model.cuda()
        print(model.__class__.__name__)

def generate_response(user_input, temperature, top_p):
    inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    return tokenizer.decode(tokens[0], skip_special_tokens=True)
def plugin_tab():
    # Streamlit UI
    st.title('Chat with StableLM')
    user_input = st.text_area("You: ", "")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.75, 0.05)
    top_p = st.slider("Top P", 0.1, 1.0, 0.95, 0.05)

    if st.button('Generate Response'):
        load_models()
        response = generate_response(user_input, temperature, top_p)
        st.text_area("Bot:", value=response, height=150, max_chars=None, key=None)

