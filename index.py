from Yt_api_call import *
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api._errors import IpBlocked, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


class Index:
    def __init__(self, url_id):
        self.url_id = url_id
        self.transcript = None
        self.vector_store = None
        self.retriever = None
        self.model = None
        self.prompt = None

    def text_splitter(self, transcript):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.create_documents([transcript])
        embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
        self.vector_store = FAISS.from_documents(chunks, embeddings)

    def retriever_engine(self):
        self.retriever = self.vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 4})

    def modelling(self):
        self.model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

    def prompting(self):
        self.prompt = PromptTemplate(
            template='''You are a helpful assistant. Answer only from the provided transcript context. If the context is insufficient, just say "I don't know"

                Context: {content}

                Question: {question}''',
            input_variables=['content', 'question']
        )

    def chaining(self, question):
        parser = StrOutputParser()

        def format_docs(retrieved_docs):
            return '\n\n'.join(doc.page_content for doc in retrieved_docs)

        parallel_chain = RunnableParallel({
            'content': self.retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        main_chain = parallel_chain | self.prompt | self.model | parser
        return main_chain.invoke(question)
