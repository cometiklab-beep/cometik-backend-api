import os
import uuid
import whisper
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- CONFIGURACIÓN DE LA API ---

app = FastAPI(
    title="COMETI-K Backend API",
    description="API para la transcripción de audio (Whisper) y análisis (Llama/LLM) de la Batería Clínica Gamificada.",
    version="1.0.0"
)

# Directorio donde se guardarán los archivos de audio y resultados
# En Codespaces o Docker, este directorio debe existir
STORAGE_DIR = "datos_clinicos" 

# Asegura que el directorio de almacenamiento exista
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# --- MODELOS DE DATOS ---

class AnalysisRequest(BaseModel):
    """Modelo para solicitar el análisis de texto por el LLM."""
    document_id: str
    pregunta_id: str
    transcription: str

class AnalysisResponse(BaseModel):
    """Modelo de la respuesta del LLM (simulada)."""
    calificacion_pragmatica: float
    comentario_llm: str
    criterios_dsm5_evaluados: dict

# --- CARGA DE MODELOS ---

# Carga el modelo de Whisper (un modelo pequeño y rápido para la nube)
# En un entorno de producción real, usarías un modelo más grande o un servicio externo
try:
    WHISPER_MODEL = whisper.load_model("base")
except Exception as e:
    print(f"Error al cargar el modelo Whisper: {e}")
    WHISPER_MODEL = None


# --- FUNCIONES DE SIMULACIÓN LLAMA ---

def simulate_llama_analysis(transcription: str) -> dict:
    """
    Simula la respuesta de un modelo LLama/Llama 3 sobre la transcripción.
    
    En un entorno real, esta función haría lo siguiente:
    1. Cargar o conectar al modelo Llama (e.g., llama-cpp-python).
    2. Enviar un prompt con la transcripción y el contexto clínico.
    3. Analizar la respuesta.
    """
    
    # Simulación de calificación y análisis
    if "no sé" in transcription.lower() or "dijo" in transcription.lower():
        score = 0.5
        comment = "La respuesta es breve o evasiva, pero contiene estructura simple. Requiere análisis contextual."
    elif len(transcription.split()) < 5:
        score = 0.3
        comment = "Respuesta muy corta. Posible dificultad en la elaboración o adaptación del discurso."
    else:
        score = 0.8
        comment = "Respuesta fluida con buena articulación y extensión. Indica competencia pragmática."

    return {
        "calificacion_pragmatica": score,
        "comentario_llm": comment,
        "criterios_dsm5_evaluados": {
            "adaptacion_interlocutor": score > 0.7,
            "normas_conversacionales": score > 0.6
        }
    }

# --- ENDPOINTS ---

@app.get("/", tags=["Estado"])
def read_root():
    """Verifica si la API está en funcionamiento."""
    return {"status": "ok", "message": "API de COMETI-K activa y lista."}

@app.post("/upload_audio/", tags=["Audio y Transcripción"])
async def upload_audio(
    document_id: str,
    pregunta_id: str,
    audio_file: UploadFile = File(...)
):
    """
    Recibe un archivo de audio, lo guarda y lo transcribe usando Whisper.
    
    Args:
        document_id: ID único del niño evaluado.
        pregunta_id: ID único de la pregunta en la batería.
        audio_file: El archivo de audio (esperado en formato compatible con Whisper/FFmpeg).
    """
    
    if not WHISPER_MODEL:
        raise HTTPException(status_code=503, detail="El modelo de transcripción Whisper no está cargado.")

    # 1. Definición de rutas y nombres de archivo
    file_extension = os.path.splitext(audio_file.filename)[1]
    
    # Crea la carpeta específica para el documento si no existe
    document_folder = os.path.join(STORAGE_DIR, document_id)
    if not os.path.exists(document_folder):
        os.makedirs(document_folder)
        
    # Nombre del archivo para guardar
    file_name = f"{pregunta_id}_{str(uuid.uuid4()).split('-')[0]}{file_extension}"
    file_path = os.path.join(document_folder, file_name)

    # 2. Guardar el archivo de audio
    try:
        with open(file_path, "wb") as buffer:
            # Lee el archivo de audio en bloques y lo escribe
            while chunk := await audio_file.read(1024 * 1024):
                buffer.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {e}")

    # 3. Transcribir el audio usando Whisper
    try:
        # Nota: Whisper automáticamente usa FFmpeg para decodificar el audio
        result = WHISPER_MODEL.transcribe(file_path)
        transcription = result["text"].strip()
    except Exception as e:
        # En caso de error de transcripción (e.g., audio corrupto o formato no soportado)
        raise HTTPException(status_code=500, detail=f"Error durante la transcripción de Whisper: {e}")

    # 4. Respuesta
    return JSONResponse(content={
        "document_id": document_id,
        "pregunta_id": pregunta_id,
        "audio_path": file_path,
        "transcription": transcription,
        "message": "Audio guardado y transcrito exitosamente. Listo para análisis."
    })

@app.post("/analyze_text/", response_model=AnalysisResponse, tags=["Análisis LLM"])
def analyze_text(request: AnalysisRequest):
    """
    Recibe una transcripción de texto y realiza un análisis simulado con Llama/LLM.
    
    Args:
        request: Objeto JSON con document_id, pregunta_id y la transcripción.
    """
    
    # 1. Realizar el análisis (simulado)
    analysis_data = simulate_llama_analysis(request.transcription)
    
    # 2. Guardar el resultado del análisis junto con los metadatos
    try:
        document_folder = os.path.join(STORAGE_DIR, request.document_id)
        if not os.path.exists(document_folder):
            os.makedirs(document_folder)
            
        analysis_filename = f"{request.pregunta_id}_analysis_{str(uuid.uuid4()).split('-')[0]}.json"
        analysis_path = os.path.join(document_folder, analysis_filename)
        
        # Datos a guardar
        full_data = {
            "document_id": request.document_id,
            "pregunta_id": request.pregunta_id,
            "transcription": request.transcription,
            "analysis": analysis_data
        }
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"Advertencia: No se pudo guardar el archivo de análisis. {e}")
        # La API aún responde, pero lanza una advertencia en la consola
        
    # 3. Respuesta
    return AnalysisResponse(**analysis_data)
  
