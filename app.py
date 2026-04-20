import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api._errors import IpBlocked, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Configure page
st.set_page_config(page_title="YouTube Q&A Assistant", page_icon="🎥", layout="wide")

# Title
st.title("🎥 YT Q&A Assistant")
st.markdown("**Powered by LangChain and Gemini LLM**")
st.markdown("---")


def extract_video_id(url_or_id):
    """Extract video ID from YouTube URL or return as-is if already an ID"""
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        if "watch?v=" in url_or_id:
            return url_or_id.split("watch?v=")[-1].split("&")[0]
        elif "youtu.be/" in url_or_id:
            return url_or_id.split("youtu.be/")[-1].split("?")[0]
    return url_or_id


def fetch_transcript_with_retry(video_id, max_retries=3):
    """Fetch transcript with retry mechanism"""
    for attempt in range(max_retries):
        try:
            transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en', 'hi'])
            # Fix: Handle both dict format and object format
            transcript_text = ""
            for chunk in transcript_list:
                if hasattr(chunk, 'text'):
                    # New format - chunk is an object with .text attribute
                    transcript_text += chunk.text + " "
                elif isinstance(chunk, dict) and 'text' in chunk:
                    # Old format - chunk is a dictionary
                    transcript_text += chunk['text'] + " "
                else:
                    # Fallback - convert to string
                    transcript_text += str(chunk) + " "
            return transcript_text

        except IpBlocked:
            if attempt == max_retries - 1:
                st.error("❌ IP blocked by YouTube. Please try again later.")
                return None
        except TranscriptsDisabled:
            st.error("❌ No captions available for this video.")
            return None
        except NoTranscriptFound:
            st.error("❌ No transcript found for the specified languages.")
            return None
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"❌ Error fetching transcript: {str(e)}")
                return None
    return None


def format_docs(retrieved_docs):

    return '\n\n'.join(doc.page_content for doc in retrieved_docs)


def process_video_and_question(video_input, question):

    if not video_input or not question:
        return "Please provide both video ID/URL and a question."

    # Extract video ID
    video_id = extract_video_id(video_input)

    try:
        # Fetch transcript
        with st.spinner("🔄 Fetching transcript..."):
            transcript = fetch_transcript_with_retry(video_id)

        if not transcript:
            return "Failed to fetch transcript. Please check the video ID and try again."

        # Process transcript
        with st.spinner("🔄 Processing transcript..."):
            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.create_documents([transcript])

            # Create vector store
            embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
            vector_store = FAISS.from_documents(chunks, embeddings)

            # Create retriever
            retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 4})

            # Setup model and prompt
            model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
            prompt = PromptTemplate(
                template='''You are a helpful assistant. Answer only from the provided transcript context. If the context is insufficient, just say "I don't know"

Context: {content}

Question: {question}''',
                input_variables=['content', 'question']
            )

            # Create chain
            parser = StrOutputParser()
            parallel_chain = RunnableParallel({
                'content': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })
            main_chain = parallel_chain | prompt | model | parser

        # Generate answer
        with st.spinner("🤔 Generating answer..."):
            answer = main_chain.invoke(question)

        return answer

    except Exception as e:
        return f"Error processing request: {str(e)}"


# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Input")

    # Video input
    video_input = st.text_input(
        "Enter YouTube Video ID or URL:",
        placeholder="e.g., PGUdWfB8nLg or https://www.youtube.com/watch?v=PGUdWfB8nLg",
        help="You can paste either a YouTube URL or just the video ID"
    )

    # Question input
    question = st.text_area(
        "Enter your question:",
        placeholder="e.g., What is the main topic of this video?",
        height=100
    )

    # Submit button
    submit_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)

with col2:
    st.header("💬 Answer")

    # Answer box (initially empty)
    answer_placeholder = st.empty()

    # Show empty state initially
    with answer_placeholder.container():
        st.info("👈 Enter a video ID/URL and question, then click 'Get Answer' to see the response here.")

# Process when button is clicked
if submit_button:
    if video_input and question:
        # Clear the placeholder and show processing
        answer_placeholder.empty()

        # Get the answer
        answer = process_video_and_question(video_input, question)

        # Display the answer in the box
        with answer_placeholder.container():
            if answer.startswith("Error") or answer.startswith("Failed") or answer.startswith("Please provide"):
                st.error(answer)
            else:
                st.success("✅ Answer generated successfully!")
                st.markdown("### 🤖 Response:")
                st.write(answer)

                # Add some additional info
                st.markdown("---")
                st.caption(f"📹 Video ID: {extract_video_id(video_input)}")
                st.caption(f"❓ Question: {question}")
    else:
        with answer_placeholder.container():
            st.warning("⚠️ Please provide both a video ID/URL and a question.")

# Add footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 14px;'>
    💡 <strong>Tips:</strong> 
    • You can use either YouTube URLs or video IDs<br>
    • Ask specific questions for better answers<br>
    • The system works best with videos that have captions
    </div>
    """,
    unsafe_allow_html=True
)

# Add example section in sidebar
with st.sidebar:
    st.header("📖 Examples")

    st.markdown("**Example Video IDs:**")
    st.code("PGUdWfB8nLg")
    st.code("dQw4w9WgXcQ")

    st.markdown("**Example Questions:**")
    st.markdown("• Who is the speaker?")
    st.markdown("• What is the main topic?")
    st.markdown("• Can you summarize this video?")
    st.markdown("• What are the key points discussed?")

    st.markdown("---")
    st.markdown("**Supported formats:**")
    st.markdown("• Video ID: `PGUdWfB8nLg`")
    st.markdown("• Full URL: `https://youtube.com/watch?v=...`")
    st.markdown("• Short URL: `https://youtu.be/...`")
