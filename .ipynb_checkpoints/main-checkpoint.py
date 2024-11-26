# importing libraries
import pandas as pd
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from functools import reduce
import nest_asyncio
import logging

from config import template, logo_path, f_path, model

# Set the logging level for httpx to WARNING to suppress INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set OpenAI API key from Streamlit secrets for security
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

nest_asyncio.apply()


# reading all the data and combining the sheets data together and cleaning and return chunks
def combine_excel_sheets(file_path):
    # Read all sheets from the Excel file
    sheets = pd.read_excel(file_path, sheet_name=None)

    # Combine data from all sheets into one column
    combined_data = []
    for sheet_name, sheet_data in sheets.items():
        # Convert each row of the sheet into a single string and add it to the list
        combined_data.extend(
            sheet_data.astype(str)
            .apply(lambda x: " ".join(x.dropna()), axis=1)
            .tolist()
        )

    # Create a DataFrame with a single column
    combined_df = pd.DataFrame(combined_data, columns=["CombinedData"])
    combined_data = " ".join(combined_df["CombinedData"].astype(str))

    replacements = {
        "\n": " ",
        "( new tab)": " ",
        "(new tab)": " ",
        "(new tab": " ",
        "Copy link": " ",
        "[Link]": " ",
    }
    combined_docs = reduce(
        lambda combined_data, kv: combined_data.replace(*kv),
        replacements.items(),
        combined_data,
    )

    # Create Document objects for each entry in the row
    all_documents = [Document(page_content=str(combined_docs))]

    # creating chunks using recursive character splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
    chunks = splitter.split_documents(all_documents)

    return chunks


def hybrid_retriever(chunks):
    # embeddings, creating vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = DocArrayInMemorySearch.from_documents(
        documents=chunks, embedding=embeddings
    )
    # add no of retrived docs for context based on similarity search
    vec_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # using bm25(bestmatch25) algo for the keyword search
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 2  # choose no of documents for keyword retriever

    # building an ensemble hybrid retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vec_retriever, keyword_retriever], weights=[0.7, 0.3]
    )

    return ensemble_retriever


# building llm chain
def llm_chain(prompt_template, model, retriever):
    prompt = PromptTemplate.from_template(prompt_template)

    # llm
    # temperature of 0 means the responses will be very straightforward and predictable, almost deterministic
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=model, temperature=0)

    # chain
    llm_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return llm_chain


# putting it all together and creating rag chain
def final_pipeline(file_path, prompt_template, model):
    text_chunks = combine_excel_sheets(file_path)
    ensemble_ret = hybrid_retriever(text_chunks)
    rag_chain = llm_chain(prompt_template, model, ensemble_ret)

    return rag_chain


def main():

    # Streamlit App Layout
    st.set_page_config(
        page_title="N26 Chatbot",
        page_icon=logo_path,
    )
    st.title("N26 Bank Chatbot")
    st.markdown(
        """
        You can ask any queries related to n26 bank.
        """
    )

    # Initialize the RAG chain (cache to prevent reloading on every interaction)
    @st.cache_resource
    def initialize_rag():
        rag_chain = final_pipeline(f_path, template, model)
        return rag_chain

    rag_chain = initialize_rag()

    # User Input
    user_question = st.text_input("Enter your question here:", "")

    # Handle the query
    if st.button("Ask"):
        if user_question.strip() != "":
            with st.spinner("Generating answer..."):
                try:
                    answer = rag_chain.invoke({"question": user_question})
                    st.success("Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a valid question.")


if __name__ == "__main__":
    main()
