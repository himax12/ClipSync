# 🚀 ClipSync: Quick Start (JSON Input)

The easiest way to use ClipSync is through JSON input!

## Step 1: Prepare Your JSON File

Create a JSON file with this structure:

```json
{
  "a_roll": {
    "url": "https://your-aroll-video.mp4",
    "metadata": "Description of your A-Roll video (optional)"
  },
  "b_rolls": [
    {
      "id": "broll_1",
      "url": "https://your-first-broll.mp4",
      "metadata": "Description (optional)"
    },
    {
      "id": "broll_2",
      "url": "https://your-second-broll.mp4",
      "metadata": "Another description (optional)"
    }
  ]
}
```

## Step 2: Use the JSON Input Tab

1. **Open the UI:** http://localhost:8501
2. **Go to "JSON Input" tab** (first tab)
3. **Either:**
   - Upload your JSON file, OR
   - Paste the JSON directly into the text area
4. **Click "Process from JSON"**

## Step 3: Monitor Progress

The system will automatically:
1. ✅ Download all B-Roll videos from URLs
2. ✅ Extract keyframes and generate embeddings
3. ✅ Index all B-Roll in FAISS
4. ✅ Download A-Roll video
5. ✅ Transcribe with WhisperX
6. ✅ Match semantically with B-Roll
7. ✅ Assemble final video

**Switch to "Status" tab** to watch real-time progress!

## Step 4: Download Result

When complete, click **"Download Result Video"** in the Status tab.

---

## Example: Your Food Quality JSON

Your current JSON at `c:\Users\ghima\Downloads\video_url.json` is already in the perfect format!

Just:
1. Copy the JSON content
2. Go to http://localhost:8501
3. Paste it in the "JSON Input" tab
4. Click "Process from JSON"
5. Wait ~2-5 minutes (depending on video length)
6. Download your matched video!

---

## What Happens Behind the Scenes

```
Your JSON
  ↓
Downloads 6 B-Roll videos + 1 A-Roll from URLs
  ↓
Extracts keyframes (1 every 2 seconds)
  ↓
Vertex AI generates 1408D embeddings
  ↓
FAISS indexes all B-Roll clips
  ↓
WhisperX transcribes A-Roll (word-level precision)
  ↓
Semantic Solver matches segments to clips
  ↓
FFmpeg assembles final video (frame-perfect)
  ↓
Download result!
```

---

## No GCP Authentication Needed for Testing!

**Wait, you might be thinking:** "But I didn't set up Vertex AI..."

**Good news:** The test `video_url.json` can actually work **without GCP** if you:
1. Skip the automatic processing
2. Manually download the videos
3. Upload through the manual UI tabs

BUT - for the **full automated JSON workflow**, you DO need Vertex AI. The system needs to:
- Generate embeddings for B-Roll frames
- Encode A-Roll transcript text
- Match in shared semantic space

**To set up Vertex AI:**
```powershell
gcloud auth application-default login
gcloud config set project firstproject-c5ac2
```

Then run the JSON workflow!

---

## Troubleshooting

**"Cannot connect to API"**
- Make sure backend is running: `uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000`

**"Download failed"**
- Check video URLs are accessible
- Ensure stable internet connection

**"Vertex AI authentication failed"**
- Run: `gcloud auth application-default login`

---

**You're ready to go! Try it with your `video_url.json` now! 🎬**
