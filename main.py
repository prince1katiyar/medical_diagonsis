
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from openai import OpenAI

app = FastAPI(title="Medical Imaging Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical-imaging-backend")

class AnalysisRequest(BaseModel):
    image: str
    filename: str
    api_key: str

@app.post("/analyze")
def analyze(request: AnalysisRequest):
    try:
        client = OpenAI(api_key=request.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages = [
            {
                "role": "system",
                "content": "You are an expert AI radiologist analyzing medical images. Your response should include a detailed radiological assessment, identifying possible conditions, underlying causes, key observations, highlights, and recommendations."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Perform a comprehensive radiological assessment of this image. "
                            "Include the following details: "
                            "1. Identified abnormalities or conditions. "
                            "2. Possible causes and contributing factors. "
                            "3. Key observations and notable findings. "
                            "4. Critical areas that require attention. "
                            "5. Recommended next steps or further evaluations. "
                            "Highlight important regions and provide an in-depth analysis."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{request.image}"}
                    }
                ]
            }
        ],
            max_tokens=2000
        )
        analysis = response.choices[0].message.content

        references_response = client.chat.completions.create(
            model="gpt-4o",
        messages = [
            {
                "role": "system",
                "content": (
                    "Provide relevant medical literature references for the conditions mentioned. "
                    "Ensure that all references are in **clickable hyperlink format** and not as plain text."
                )
            },
            {
                "role": "user",
                "content": analysis
            }
        ],

            max_tokens=1000
        )
        references = references_response.choices[0].message.content

        return {"analysis": analysis, "references": references}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
