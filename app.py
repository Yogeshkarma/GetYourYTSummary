import streamlit as st
from Yt_api_call import Youtube_Fetcher
from index import Index
from dotenv import load_dotenv

load_dotenv()


class App:
    def __init__(self):
        st.set_page_config(page_title="YouTube Q&A Assistant", page_icon="🎥", layout="wide")

    def format_docs(self, retrieved_docs):
        return '\n\n'.join(doc.page_content for doc in retrieved_docs)

    def process_video_and_question(self, video_input, question):
        if not video_input or not question:
            return "Please provide both video ID/URL and a question."

        try:
            # Use Youtube_Fetcher class
            with st.spinner("🔄 Fetching transcript..."):
                fetcher = Youtube_Fetcher(video_input)
                transcript = fetcher.fetch_transcript()

            if not transcript:
                return "Failed to fetch transcript. Please check the video ID and try again."

            # Use Index class to process
            with st.spinner("🔄 Processing transcript..."):
                processor = Index(video_input)
                processor.text_splitter(transcript)
                processor.retriever_engine()
                processor.modelling()
                processor.prompting()

            # Generate answer
            with st.spinner("🤔 Generating answer..."):
                answer = processor.chaining(question)

            return answer

        except Exception as e:
            return f"Error processing request: {str(e)}"

    def run(self):
        # Title
        st.title("🎥 YT Q&A Assistant")
        st.markdown("**Powered by LangChain and Gemini LLM**")
        st.markdown("---")

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
                answer = self.process_video_and_question(video_input, question)

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
                        fetcher = Youtube_Fetcher(video_input)
                        st.caption(f"📹 Video ID: {fetcher.url_validator()}")
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
