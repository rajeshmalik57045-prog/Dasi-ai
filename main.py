import os, base64, io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv
import edge_tts

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """तुम जार्विस हो — स्मार्ट, तेज़, friendly AI assistant।
हिंदी/हिंग्लिश में जवाब दो। Technical शब्द English में रखो।
छोटे, natural जवाब दो। किताबी हिंदी नहीं।"""

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/chat")
async def chat(message: str = Form(...)):
    def generate():
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            stream=True, max_tokens=1024,
        )
        for chunk in stream:
            d = chunk.choices[0].delta.content
            if d: yield d
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/chat-image")
async def chat_image(message: str = Form(...), image: UploadFile = File(...)):
    data = await image.read()
    if len(data) > 4 * 1024 * 1024:
        return {"error": "फोटो 4MB से छोटी भेजो"}
    b64 = base64.b64encode(data).decode()
    r = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": message},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}], max_tokens=1024,
    )
    return {"reply": r.choices[0].message.content}

@app.post("/speak")
async def speak(text: str = Form(...)):
    communicate = edge_tts.Communicate(text, "hi-IN-MadhurNeural")
    audio = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.write(chunk["data"])
    audio.seek(0)
    return StreamingResponse(audio, media_type="audio/mpeg")
