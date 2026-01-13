"""
Streamlit Frontend - Semantic A-Roll/B-Roll Engine
"""

import streamlit as st
import requests
from pathlib import Path
import time
import json

# Configuration
API_BASE = "http://localhost:8000/api"

# Page config
st.set_page_config(
    page_title="Semantic A-Roll/B-Roll Engine",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎬 Semantic Video Matching Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered A-Roll/B-Roll Matching with Vertex AI + WhisperX + FAISS</div>', unsafe_allow_html=True)

# Check API health (backend can take 10+ seconds to initialize)
api_online = False
with st.spinner("🔄 Connecting to backend API..."):
    try:
        response = requests.get(f"{API_BASE.replace('/api', '')}/", timeout=15)
        if response.ok:
            st.success("✅ Backend API is online")
            api_online = True
        else:
            st.error(f"❌ Backend API returned error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API. Please start the FastAPI server.")
        st.code("uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload", language="bash")
    except requests.exceptions.Timeout:
        st.warning("⏳ Backend API is starting up... Please refresh the page in a few seconds.")
    except Exception as e:
        st.error(f"❌ Backend API error: {e}")

if not api_online:
    st.stop()

# Get index stats
try:
    stats_response = requests.get(f"{API_BASE}/index/stats")
    if stats_response.ok:
        index_stats = stats_response.json()
    else:
        index_stats = {"num_clips": 0, "total_vectors": 0}
except:
    index_stats = {"num_clips": 0, "total_vectors": 0}

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 JSON Input", "📤 Upload B-Roll", "🎥 Process A-Roll", "📊 Status", "ℹ️ About"])

# ============================================================================
# TAB 1: JSON Input (NEW!)
# ============================================================================
with tab1:
    st.header("🚀 Quick Start with JSON")
    
    st.info("""
    **Upload a JSON file** with your A-Roll and B-Roll video URLs and let the system handle everything!
    
    Expected JSON format:
    ```json
    {
      "a_roll": {"url": "https://...", "metadata": "..."},
      "b_rolls": [
        {"id": "broll_1", "url": "https://...", "metadata": "..."},
        ...
      ]
    }
    ```
    """)
    
    # JSON Upload
    uploaded_json = st.file_uploader(
        "Upload JSON File",
        type=["json"],
        help="Upload a JSON file containing video URLs"
    )
    
    # Or paste JSON directly
    st.markdown("**Or paste JSON directly:**")
    json_text = st.text_area(
        "Paste JSON here",
        height=200,
        placeholder='{"a_roll": {"url": "..."}, "b_rolls": [...]}'
    )
    
    if st.button("🚀 Process from JSON", type="primary"):
        import json
        
        # Parse JSON
        try:
            if uploaded_json:
                json_data = json.load(uploaded_json)
            elif json_text:
                json_data = json.loads(json_text)
            else:
                st.error("Please upload a JSON file or paste JSON text")
                st.stop()
            
            st.success("✅ JSON parsed successfully!")
            
            # Display what was found
            if "a_roll" in json_data:
                st.write(f"**A-Roll:** {json_data['a_roll'].get('metadata', 'No description')[:80]}...")
            
            if "b_rolls" in json_data:
                st.write(f"**B-Rolls:** {len(json_data['b_rolls'])} videos found")
            
            # Send to backend
            with st.spinner("🔄 Sending to backend for processing..."):
                response = requests.post(
                    f"{API_BASE}/process/json",
                    json=json_data,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.ok:
                    result = response.json()
                    job_id = result.get("job_id")
                    st.session_state["job_id"] = job_id
                    st.session_state["start_time"] = time.time()
                    
                    st.success(f"🎉 Processing started! Job ID: `{job_id}`")
                    st.info("📊 Switch to the **Status** tab to monitor progress")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ Processing failed: {response.text}")
                    
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# TAB 2: Upload B-Roll (Manual upload - original tab)
# ============================================================================
with tab2:
    st.header("B-Roll Library Management")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Indexed Clips", index_stats.get("num_clips", 0))
    with col2:
        st.metric("Total Vectors", index_stats.get("total_vectors", 0))
    with col3:
        st.metric("Dimension", index_stats.get("dimension", 1408))
    
    st.markdown("---")
    
    st.info("💡 **Tip:** Upload multiple B-Roll clips that cover different topics. The AI will select the most semantically relevant clips for each A-Roll segment.")
    
    uploaded_files = st.file_uploader(
        "Upload B-Roll clips",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=True,
        help="Upload video files to be indexed in the B-Roll library"
    )
    
    if st.button("🚀 Index B-Roll Library", disabled=not uploaded_files, type="primary"):
        with st.spinner("Processing B-Roll clips... This may take a few minutes."):
            # Prepare files for upload
            files = [("files", (f.name, f, "video/mp4")) for f in uploaded_files]
            
            try:
                response = requests.post(f"{API_BASE}/upload/broll", files=files)
                
                if response.ok:
                    data = response.json()
                    st.success(f"✅ Successfully indexed {data['clips_indexed']} clips in {data['processing_time']:.1f}s!")
                    
                    # Show stats
                    new_stats = data.get("index_stats", {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Clips", new_stats.get("num_clips", 0))
                    with col2:
                        st.metric("Processing Time", f"{data['processing_time']:.1f}s")
                    
                    st.rerun()
                else:
                    st.error(f"❌ Upload failed: {response.text}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# TAB 3: Process A-Roll
# ============================================================================
with tab3:
    st.header("Process A-Roll Video")
    
    if index_stats.get("num_clips", 0) == 0:
        st.warning("⚠️ No B-Roll clips indexed yet. Please upload B-Roll clips first in the 'Upload B-Roll' tab.")
    else:
        st.info(f"✅ Ready to process! {index_stats['num_clips']} B-Roll clips available for matching.")
    
    st.markdown("---")
    
    aroll_file = st.file_uploader(
        "Upload A-Roll video",
        type=["mp4", "mov", "avi"],
        help="Upload your main video with narration/dialogue"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        k_candidates = st.slider("Candidate Pool Size", 1, 10, 5, help="Number of B-Roll candidates to consider per segment")
    with col2:
        min_similarity = st.slider("Minimum Similarity", 0.0, 1.0, 0.0, 0.1, help="Minimum semantic similarity threshold")
    
    if st.button(
        "🚀 Start Processing",
        disabled=not aroll_file or index_stats.get("num_clips", 0) == 0,
        type="primary"
    ):
        files = {"aroll": (aroll_file.name, aroll_file, "video/mp4")}
        
        try:
            response = requests.post(f"{API_BASE}/process", files=files)
            
            if response.ok:
                job_id = response.json()["job_id"]
                st.session_state["job_id"] = job_id
                st.session_state["start_time"] = time.time()
                st.success(f"✅ Processing started! Job ID: `{job_id}`")
                st.info("Switch to the 'Status' tab to monitor progress")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ Failed to start processing: {response.text}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# TAB 4: Status
# ============================================================================
with tab4:
    st.header("Processing Status")
    
    if "job_id" in st.session_state:
        job_id = st.session_state["job_id"]
        
        # Auto-refresh while processing
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        message_placeholder = st.empty()
        stats_placeholder = st.empty()
        download_placeholder = st.empty()
        
        try:
            response = requests.get(f"{API_BASE}/status/{job_id}")
            
            if response.ok:
                data = response.json()
                status = data.get("status")
                progress = data.get("progress", 0)
                message = data.get("message", "")
                
                # Status indicator
                if status == "processing" or status == "queued":
                    status_placeholder.info(f"⏳ Status: **{status.upper()}**")
                elif status == "complete":
                    status_placeholder.success(f"✅ Status: **{status.upper()}**")
                elif status == "error":
                    status_placeholder.error(f"❌ Status: **{status.upper()}**")
                
                # Progress bar
                progress_placeholder.progress(progress / 100)
                message_placeholder.write(f"**Message:** {message}")
                
                # Stats
                if status == "complete":
                    stats = data.get("stats", {})
                    if stats:
                        st.markdown("### 📊 Matching Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Segments", stats.get("total_segments", 0))
                        with col2:
                            st.metric("Matched Segments", stats.get("matched_segments", 0))
                        with col3:
                            match_rate = stats.get("match_rate", 0) * 100
                            st.metric("Match Rate", f"{match_rate:.1f}%")
                        with col4:
                            avg_sim = stats.get("avg_similarity", 0)
                            st.metric("Avg Similarity", f"{avg_sim:.3f}")
                    
                    #Download button
                    elapsed = time.time() - st.session_state.get("start_time", time.time())
                    st.success(f"🎉 Processing complete in {elapsed:.1f} seconds!")
                    
                    if st.button("💾  Download Result Video", type="primary"):
                        download_url = f"{API_BASE}/download/{job_id}"
                        st.markdown(f"[Click here to download]({download_url})")
                
                # Auto-refresh if processing
                if status in ["processing", "queued"]:
                    time.sleep(2)
                    st.rerun()
                    
            else:
                st.error(f"❌ Failed to get status: {response.text}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👈 No active processing job. Start one in the 'Process A-Roll' tab.")

# ============================================================================
# TAB 5: About
# ============================================================================
with tab5:
    st.header("About This System")
    
    st.markdown("""
    ## 🎯 The Semantic Optimization Engine
    
    This system maximizes the semantic correlation between Audio (A-Roll) and Video (B-Roll) 
    using state-of-the-art AI models.
    
    ### 🏗️ Architecture
    
    **Phase 1: Audio Sensor (A-Roll Processing)**
    - **WhisperX**: Word-level transcription with phoneme alignment
    - **Wav2Vec2**: Frame-accurate timestamp alignment
    
    **Phase 2: Vision Sensor (B-Roll Processing)**
    - **Vertex AI Multimodal Embeddings**: 1408-dimensional semantic vectors
    - **FFmpeg**: Efficient keyframe extraction (I-frames @ 0.5fps)
    
    **Phase 3: Memory Layer (Indexing)**
    - **FAISS GPU**: Sub-millisecond similarity search
    - **Inner Product**: Cosine similarity via normalized vectors
    
    **Phase 4: Solver (Matching Algorithm)**
    - **Greedy Algorithm**: With lookahead and constraint validation
    - **Constraints**: Duration matching, clip distinctness, similarity threshold
    
    **Phase 5: Actuator (Video Assembly)**
    - **FFmpeg Filter Complex**: Single-pass frame-perfect rendering
    - **Audio Passthrough**: Lossless original audio preservation
    
    ### ⚡ Performance
    
    - **Target**: Process 10-minute video in <2 minutes
    - **Search Speed**: 100k clips in <1ms on GPU
    - **Embedding Space**: 1408D unified multimodal space
    
    ### 🔧 Tech Stack
    
    - **Backend**: FastAPI + UV
    - **Frontend**: Streamlit
    - **AI Models**: Vertex AI, WhisperX
    - **Search**: FAISS GPU
    - **Video**: FFmpeg
    
    ### 📝 Configuration
    
    Edit `.env` file to configure:
    - GCP Project ID
    - GPU settings
    - Processing parameters
    """)
    
    st.markdown("---")
    st.caption("Built with ❤️ using First Principles Thinking")

