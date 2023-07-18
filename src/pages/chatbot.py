# https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads
# https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
# https://zenn.dev/umi_mori/books/prompt-engineer/viewer/chatgpt_webapp_langchain_streamlit
# https://zenn.dev/umi_mori/books/prompt-engineer/viewer/langchain_memory
# https://ainow.ai/2022/08/30/267101/
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import torch

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain import PromptTemplate, LLMChain, HuggingFacePipeline
from langchain.schema import HumanMessage
from langchain.schema import AIMessage


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
        # "RWKV/rwkv-4-169m-pile",
        # "RWKV/rwkv-4-430m-pile",
        "RWKV/rwkv-raven-1b5",
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
        "abeja/gpt-neox-japanese-2.7b",
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


def get_template(model_id):
    if model_id == chatgpt_id or model_id in en_list:
#         temp = """You are a chatbot having a conversation with a human.
# {chat_history}
# Human: {input}
# Chatbot: """
        temp = '''{chat_history}
Human: {input}
AI: Let's think step by step. '''
    else:
        temp = '''{chat_history}
質問: {input}
回答: Let's think step by step. '''
    placeholder = """Input prompt templete.
[Example]
{chat_history}
Human: {input}
AI: Let's think step by step. """
    template = st.text_area("Enter prompt template. Please insert :red[{chat_history}] and :red[{input}].", temp,
                           placeholder=placeholder)
    return template


def get_prompt(template):
    prompt = PromptTemplate.from_template(template)
    return prompt


def chatbot():
    # Streamlitによって、タイトル部分のUIをの作成
    st.title("Chatbot :robot_face::left_speech_bubble::grinning:")
    st.caption("Running on "+(":green[cuda]" if torch.cuda.is_available() else ":green[cpu]"))

    st.header("Language Model selection")
    model_id = st.selectbox(
        "Choose language model (LM)",
        (
            *tuple(en_list),
            *tuple(ja_list),
            chatgpt_id
        )
    )

    with st.expander("Details of LM"):
        st.markdown("""#### 英語または多言語
- gpt2
    - Transformerをベースにし自己教師あり事前学習、ファインチューニングした大規模言語モデル(Large Language Model; LLM)であるGPTの発展形
        - 事前学習は、文章をわざと欠損させた部分を補完するように推定するタスク
        - 自己回帰モデルというものであり、文脈を考慮するように帰納バイアスを持つ
    - パラメータ数と学習データ数が増加した
        - 先代であるGPTは4.5GBのテキスト（書籍等）
        - GPT2は40GBのテキスト（大量のウェブページ）
- bloom
    - 多言語のLLM
    - 46の自然言語と13のプログラミング言語のデータセットで学習
    - オープンソースであり、多くの研究者の協力によりつくられた
- bloomz
    - 上記bloomを多言語とそのタスクでの汎化性能の向上を、英語プロンプトと英語タスクでのファインチューニングによって達成したもの
- opt
    - Meta社が取り組んでいる研究者向けのLLM
    - 125Mから175Bのパラメータを持つデコーダのみの事前学習されたもの
- Cerebras-GPT
    - Chinchillaを参考にして学習されたモデル
    - 111Mから13Bのパラメータ
    - AIアクセラレータの会社であるCerebrasが作成している
    - オープンアクセス
- gpt2-alpaca
    - ChatGPTとの会話データでファインチューニングするモデルをAlpacaという
    - gpt2のalpacaということ
#### 日本語
- open-calm
    - GPT-NeoXを基に日本語のWikipediaやWebページで学習したモデル
    - サイバーエージェントが作成している
- rinna/japanese-gpt-neox-small
    - GPT-NeoXを基に日本語のWikipediaやコーパスで学習したモデル
    - rinnaが作成している
- abeja/~
    - GPT-NeoXやGPT2を基に日本語のWikipediaやコーパスで学習したモデル
    - abejaが作成している
        
""")


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
        
        
        st.subheader("Memory window size")
        mem_window = st.slider("Memory window size", 0, 10, 5)
        
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
    st.header("Prompt configuration")
    template = get_template(model_id)
    prompt = get_prompt(template)
    
    if "prompt_temp" not in st.session_state:
        st.session_state["prompt_temp"] = template
    if st.session_state["prompt_temp"] != template:
        st.session_state["prompt_temp"] = template
        memory = ConversationBufferWindowMemory(
                                                memory_key="chat_history",
                                                k=mem_window
        )
        st.session_state["memory"] = memory

    # セッション内に保存されたチャット履歴のメモリの取得
    try:
        memory = st.session_state["memory"]
    except:
        memory = ConversationBufferWindowMemory(
                                                memory_key="chat_history",
                                                k=5
        )

    chain = LLMChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )

    st.header("Let's chat!!")
    # 入力フォームと送信ボタンのUIの作成
    left, right = st.columns([8, 2])
    with left:
        # text_input = st.chat_input("Input texts.")
        text_input = st.text_area("Enter your message",
                                  "What is AI?" if model_id in en_list else "AIとは何？",
                                  placeholder="Input texts.")
    with right:
        st.markdown("")
        st.markdown("")
        exe = st.button("Run")
        clear = st.button("Clear")

    if clear:
        clear = False
        try:
            memory.clear()
            st.session_state["memory"] = memory
            text_input = ""
        except Exception as e:
            st.error(e)

    # チャット履歴（HumanMessageやAIMessageなど）を格納する配列の初期化
    history = []

    if len(text_input)>0 or exe:
        exe = False
        # ChatGPTの実行
        chain(text_input)
        # chain.predict(input=text_input)

        # セッションへのチャット履歴の保存
        st.session_state["memory"] = memory

        # チャット履歴（HumanMessageやAIMessageなど）の読み込み
        try:
            history = memory.buffer
        except Exception as e:
            st.error(e)

    # チャット履歴の表示
    for index, chat_message in enumerate(reversed(history)):
        if type(chat_message) == HumanMessage:
            message(chat_message.content, is_user=True, key=2 * index)
        elif type(chat_message) == AIMessage:
            message(chat_message.content, is_user=False, key=2 * index + 1)

    # st.write(memory.load_memory_variables({})["chat_history"])

# if __name__ == "main":
chatbot()
