import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain import PromptTemplate
from langchain import OpenAI, LLMChain
from langchain.chains import ConversationalRetrievalChain

PINECONE_INDEX_NAME = 'pdf-test-index'

PINECONDE_NAME_SPACE = 'dl-pdf'

pinecone.init(api_key="10e2163e-b3a2-40f4-b5df-1ff98600b76b",
              environment="us-east4-gcp")

CONDENSE_PROMPT = PromptTemplate(input_variables=['chat_history', 'question'], template='''Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:''')

QA_PROMPT = PromptTemplate(input_variables=['question', 'context'], 
template='''You are an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.
You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks.
If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
Question: {question}
=========
{context}
=========
Answer in Markdown:''')


async def ingest_data():
    loader = PyPDFLoader("2014-765.pdf")
    rawDocs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 100,
        chunk_overlap  = 20,
        length_function = len,
    )

    docs = text_splitter.split_documents(rawDocs)
    print('split docs', docs)

    embeddings = OpenAIEmbeddings()

    index = pinecone.Index(PINECONE_INDEX_NAME)

    docsearch = Pinecone.from_documents(docs, embeddings, index_name=PINECONE_INDEX_NAME)

    pass

def make_chain(vectorstore):

    llm = OpenAI(temperature=0.5)

    question_generator = LLMChain(
        llm=llm,
        prompt=CONDENSE_PROMPT
    )

    doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

    return ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(),
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    )


async def run():
    chat_history = []

    embeddings = OpenAIEmbeddings()

    vectorstore = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings, "text")

    chain = make_chain(vectorstore)
    
    continue_qa = True

    while continue_qa == True:
        user_question = input("GPT-PDF: ")
        response = chain({"question": user_question, "chat_history": chat_history})
        print(f'\n{response["answer"]}\n')
        chat_history.append([response['question'], response["answer"]])
    
    
    pass

if __name__ == '__main__':
    # asyncio.run(ingest_data())
    asyncio.run(run())