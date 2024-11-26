N26 Chatbot can be aacessed here: https://kiran-n26chatbot.streamlit.app/

Built a Customer Service chatbot for the N26 customers to help them with their queries
 - All the data is taken from N26 Bank's faq support pages: https://support.n26.com/en-eu

- Built a rag system using hybrid search retrieve the best possible documents matched with the user query
- OpenAI and Langchain frameworks are used
- To enhance the retrieval process in RAG, Hybrid Search (keyword search and vector search) appproach is used
  - for keyword search BM25 algorithm is used
  - for vectorstore DocArrayInMemorySearch is used while embedding from OpenaAI are used (one cam also use chroma or other available vectore stores)
  
- For evaluation of the RAG pipeline, ragas library is used https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
  - These metrics are used to evaluate the rag pipeline (these metrics are llm based)
    - answer_relevancy (text generation)
    - faithfulness (text generation)
    - context_recall (text retrieval)
    - context_precision (text retrieval)
