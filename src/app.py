import streamlit as st


def main():
    st.title("Welcome to LM examples!!")
    
    st.header("Chatbot")
    st.write("##### チャットボットです。対話を楽しむことができます。")
    st.image("./pages/imgs/chatbot.png")
    
    st.header("ChatPDF")
    st.write("##### PDFを読み込み、そのPDFの内容を踏まえた対話ができます。")
    st.image("./pages/imgs/chatpdf.png")
    
    st.subheader("プログラムへのリンク")
    st.image("./pages/imgs/QR.png")
    

if __name__ == "__main__":
    main()
