# 🚀 ClipSync: Google Colab Setup Guide

## Why Use Colab?

**Your Problem:** Windows crashes with `ntdll.dll` errors  
**Colab Solution:** Free GPU (Tesla T4), stable Linux, no local compute needed!

---

## Quick Setup (5 minutes)

### Step 1: Open Notebook in Colab

1. Go to https://colab.research.google.com/
2. Click **File → Upload notebook**
3. Upload `colab_backend.ipynb` from this folder

### Step 2: Enable GPU

1. Click **Runtime → Change runtime type**
2. Select **GPU** (T4)
3. Click **Save**

### Step 3: Get Ngrok Token

1. Go to https://dashboard.ngrok.com/signup (free account)
2. Copy your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
3. Paste it in Cell 5 of the notebook

### Step 4: Prepare Backend Code

Zip your `backend` folder:
```powershell
Compress-Archive -Path backend -DestinationPath backend.zip
```

### Step 5: Run Cells in Order

Run each cell from 1-8 in the notebook:
- Cell 1: Installs dependencies (~2 min)
- Cell 2: Upload backend.zip
- Cell 3: Authenticate with GCP
- Cell 4: Create directories
- Cell 5: Configure ngrok
- Cell 6: Start FastAPI
- Cell 7: Get public URL
- Cell 8: Test with JSON

---

## Connect Local Frontend

After Cell 7, you'll get a URL like: `https://abc123.ngrok.io`

**Update your local Streamlit:**

Edit `frontend/app.py` line 10:
```python
API_BASE = "https://abc123.ngrok.io"  # ← Your ngrok URL
```

Restart Streamlit:
```powershell
uv run streamlit run frontend/app.py
```

**Now your frontend (local) talks to backend (Colab GPU)!**

---

## Advantages

✅ **Free GPU:** Tesla T4 (16GB VRAM) - way better than GTX 1650  
✅ **Stable:** Linux environment, no ntdll.dll crashes  
✅ **Fast:** WhisperX on GPU = 10x faster  
✅ **No PyTorch Issues:** Colab has compatible versions  
✅ **12 hours free:** Reconnect if session expires  

---

## Performance Expectations

**Your 1-min video on Colab GPU:**
- Download videos: 10s
- Embed B-Roll: 15s (GPU fast!)
- WhisperX: 5s (GPU!)
- Match + Assemble: 30s

**Total: ~60 seconds** (vs your local 5+ min with crashes!)

---

## Troubleshooting

**"Runtime disconnected"**  
→ Run the keep-alive cell at the end

**"GCP authentication failed"**  
→ Make sure you ran Cell 3 and approved the popup

**"ngrok connection refused"**  
→ Verify your authtoken is correct

**"Module not found"**  
→ Restart runtime and run Cell 1 again

---

## Download Results

After processing completes, download from:
```
https://your-ngrok-url.ngrok.io/api/download/JOB_ID
```

Or use the Streamlit UI!

---

**Ready to try? Upload the notebook to Colab and run it!** 🚀
