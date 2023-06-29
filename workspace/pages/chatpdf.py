# https://dev.classmethod.jp/articles/python-parse-pdf/
# https://colab.research.google.com/github/nyanta012/demo/blob/main/sentence_retrieval.ipynb#scrollTo=_5bY_6TK_yFC
# https://blog.langchain.dev/langchain-chat/
# https://zenn.dev/umi_mori/books/prompt-engineer/viewer/langchain_indexes
import streamlit as st
from streamlit_chat import message
import torch
from dotenv import load_dotenv
import fitz

from langchain.chat_models import ChatOpenAI
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document


chatgpt_id = "gpt-3.5-turbo"
en_list = [
        "bigscience/bloom-560m",
        "bigscience/bloom-1b7",
        "bigscience/bloomz-560m",
        "bigscience/bloomz-1b7",
        # "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        # "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        # "cerebras/Cerebras-GPT-111M",
        "cerebras/Cerebras-GPT-256M",
        "cerebras/Cerebras-GPT-590M",
        "cerebras/Cerebras-GPT-1.3B",
        "vicgalle/gpt2-alpaca",
]
ja_list = [
        # "cyberagent/open-calm-small",
        "cyberagent/open-calm-medium",
        "cyberagent/open-calm-large",
        "cyberagent/open-calm-1b",
        # "rinna/japanese-gpt2-xsmall",
        # "rinna/japanese-gpt2-small",
        "rinna/japanese-gpt2-medium",
        # "rinna/japanese-gpt-1b",
        "rinna/japanese-gpt-neox-small",
        "abeja/gpt2-large-japanese",
        # "abeja/gpt-neox-japanese-2.7b",
]


def get_llm(model_id, model_kwargs, pipeline_kwargs):
    if torch.cuda.is_available():
        device = 0
    else:
        device = -1
    if model_id == chatgpt_id:
        load_dotenv()
        llm = ChatOpenAI(model_name=chatgpt_id)
    else:
        llm = HuggingFacePipeline.from_model_id(
            model_id, task="text-generation",
            model_kwargs=model_kwargs,
            pipeline_kwargs=pipeline_kwargs,
            device=device,
            verbose=True
        )

    return llm


def get_embeddings(model_id):
    if model_id == chatgpt_id:
        return OpenAIEmbeddings()
    # elif model_id in ja_list:
    #     return HuggingFaceEmbeddings(model_name="oshizo/sbert-jsnli-luke-japanese-base-lite")
    else:
        return HuggingFaceEmbeddings()

def chatpdf():
    st.title("ChatPDF :memo::robot_face::left_speech_bubble:")
    st.caption("Running on "+(":green[cuda]" if torch.cuda.is_available() else ":green[cpu]"))
    
    st.header("Language Model selection")
    model_id = st.selectbox(
        "Choose language model (LM)",
        (
            *tuple(en_list),
            *tuple(ja_list),
            # chatgpt_id
        )
    )
    
    with st.expander("Details of ChatPDF"):
        st.image("https://blog.langchain.dev/content/images/2023/01/Screen-Shot-2023-01-16-at-10.16.00-PM.png")
        st.image("https://blog.langchain.dev/content/images/2023/03/image-1.png")
    
    with st.sidebar:
        st.subheader("Model parameters")
        min_length = st.slider("min length", 0, 20, 20)
        max_length = st.slider("max length", 20, 200, 100)
        min_new_tokens = st.slider("min new tokens", 0, 25, 5)
        max_new_tokens = st.slider("max new tokens", 25, 100, 50)
        repetition_penalty = st.slider("repetition penalty", 1.0, 1.1, 1.01)
        do_sample = st.checkbox("do sample", True)
        top_p = st.slider("top p", 0.0, 1.0, 0.95)
        top_k = st.slider("top k", 0, 50, 50)
        temperature = st.slider("temperature", 0.01, 1.0, 0.1)
        # num_beams = st.slider("num beams", 1, 5, 1)
        # no_repeat_ngram_size = st.slider("no repeat ngram size", 0, 5, 0)
        # early_stopping = st.checkbox("early stopping", False)
        
    model_kwargs = {
                "min_length": min_length,
                "max_length": max_length,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                # "num_beams": num_beams,
                # "no_repeat_ngram_size": no_repeat_ngram_size,
                # "early_stopping": early_stopping
            }

    pipeline_kwargs = {
                "min_new_tokens": min_new_tokens,
                "max_new_tokens": max_new_tokens,
    }
    
    with st.spinner("Loading LLM..."):
        llm = get_llm(model_id, model_kwargs, pipeline_kwargs)
    
    st.header("Upload a PDF file.")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    if uploaded_file is not None:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        
        docs = [
                    Document(
                        page_content=page.get_text().encode("utf-8"),
                        metadata=dict(
                            {
                                "page_number": page.number + 1,
                                "total_pages": len(doc),
                            },
                            **{
                                k: doc.metadata[k]
                                for k in doc.metadata
                                if type(doc.metadata[k]) in [str, int]
                            },
                        ),
                    )
                    for page in doc
                ]
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)
        # st.write(texts)
    
        embeddings = get_embeddings(model_id)
        vectordb = Chroma.from_documents(texts, embeddings)
        
        qa_chain = pdf_qa = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(), return_source_documents=True)
        
        st.header("Let's chat!!")
        left, right = st.columns([8, 2])
        with left:
            # text_input = st.chat_input("Input texts.")
            text_input = st.text_area("Enter your message",
                                      "Tell me some details.",
                                      placeholder="Input texts.")

        with right:
            st.markdown("")
            st.markdown("")
            exe = st.button("Run")
            clear = st.button("Clear")

        try:
            chat_history = st.session_state["pdf_history"]
        except:
            chat_history = []
            
        if clear:
            clear = False
            try:
                chat_history = []
                st.session_state["pdf_history"] = chat_history
                text_input = ""
            except Exception as e:
                st.error(e)
        
        if len(text_input)>0 or exe:
            exe = False
            result = qa_chain({"question": text_input, "chat_history": chat_history})
            chat_history.append((text_input, result["answer"], result["source_documents"]))
            st.session_state["pdf_history"] = chat_history
            # st.write(result["source_documents"])
            
        for index, chat_message in enumerate(reversed(chat_history)):
            message(chat_message[1], is_user=False, key=2*index+1)
            message(chat_message[0], is_user=True, key=2*index)
        

chatpdf()