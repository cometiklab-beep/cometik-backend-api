import os
import uuid
import json
import re 
import csv 
from datetime import datetime
from pathlib import Path
import tempfile # Usaremos esto para el audio

# Importaciones de librerías
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Intenta importar los modelos grandes (Whisper/SQLAlchemy)
# Si falla, el servidor sigue vivo, pero las funciones asociadas no funcionarán.
try:
    import whisper
    from sqlalchemy import create_engine, text 
    # Usaremos el driver psycopg2 para PostgreSQL, aunque Render lo inyecta
except ImportError as e:
    print(f"❌ ADVERTENCIA CRÍTICA: Faltan módulos clave (Whisper/SQLAlchemy). Las funciones fallarán. {e}")
    whisper = None # Marca como no disponible
    create_engine = None # Marca como no disponible

# --- CONFIGURACIÓN DE LA API ---

API_DESCRIPTION = (
    "API para la recolección, transcripción y análisis automatizado de respuestas verbales infantiles "
    "en el marco de la validación convergente entre el CCC-2 y COMETI-K. Esta batería cognitivo-lingüística "
    "gamificada en Unity. Versión para Render (Simulación de LLama)."
)

app = FastAPI(
    title="COMETI-K Backend Clínico y Lingüístico",
    description=API_DESCRIPTION,
    version="1.1.0"
)

# Directorio de almacenamiento de datos clínicos
# Usaremos una ruta absoluta confiable en el sistema de archivos de Render (o /tmp para archivos temporales)
# Render no garantiza que `STORAGE_DIR` persista entre deploys, pero es bueno para la ejecución.
STORAGE_DIR = Path("/opt/render/project/src/datos_clinicos") 
if not STORAGE_DIR.exists():
    STORAGE_DIR.mkdir()

# --- CARGA DE MODELOS Y BD (Global, para que cargue UNA SOLA VEZ al inicio) ---

# Carga del Modelo Whisper
WHISPER_MODEL = None
try:
    if whisper:
        # Usa el modelo más pequeño para fiabilidad en Render
        WHISPER_MODEL = whisper.load_model("tiny") 
        print("✅ Modelo Whisper 'tiny' cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo Whisper: {e}. La transcripción no estará disponible.")
    WHISPER_MODEL = None

# Configuración de la Base de Datos
DATABASE_URL = os.environ.get('DATABASE_URL')
DB_ENGINE = None
if DATABASE_URL and create_engine:
    try:
        # Crea el motor de conexión a la base de datos
        DB_ENGINE = create_engine(DATABASE_URL)
        print("✅ Motor de PostgreSQL configurado correctamente.")
    except Exception as e:
        print(f"❌ Error al crear el motor de PostgreSQL: {e}. El guardado en BD no estará disponible.")

# LLAMA SIMULACIÓN (Mantenemos la simulación para evitar el timeout del modelo GGUF grande)
LLAMA_MODEL = None
print("⚠️ Modo de Análisis de LLama forzado a SIMULACIÓN para evitar Timeouts. El modelo LLama no fue cargado.")

# --- MODELOS DE DATOS ---
class AnalysisRequest(BaseModel):
    document_id: str
    pregunta_id: str
    transcription: str

class AnalysisResponse(BaseModel):
    # CRITERIOS
    calificacion_pragmatica_dsm5: float  
    calificacion_pragmatica_ampliada: float
    comentario_llm: str
    puntuacion_a1_uso_social: int
    puntuacion_a2_ajuste_contexto: int
    puntuacion_a3_normas_conversacionales: int
    puntuacion_a4_comprension_no_literal: int
    puntuacion_a5_coherencia: int          
    puntuacion_a6_cohesion: int            
    analisis_complejidad_sintactica: int
    analisis_disfluencias: int

# --- FUNCIONES DE SIMULACIÓN Y ANÁLISIS ---

def simulate_llama_analysis(transcription: str) -> dict:
    """Función de análisis SIMULADO, con las 8 métricas requeridas."""
    # (Tu lógica de simulación, que está bien, se mantiene)
    word_count = len(transcription.split())
    
    if word_count < 5:
        puntuaciones_dsm5 = [0, 0, 1, 0] 
        puntuaciones_discurso = [0, 1]  
        comment = "SIMULADO: Respuesta muy corta. Baja evidencia de competencia pragmática."
    elif word_count < 15:
        puntuaciones_dsm5 = [1, 1, 1, 0]
        puntuaciones_discurso = [1, 1]
        comment = "SIMULADO: Respuesta coherente, pero breve. Competencia media."
    else:
        puntuaciones_dsm5 = [2, 2, 2, 1]
        puntuaciones_discurso = [2, 2]
        comment = "SIMULADO: Respuesta fluida con buena articulación. Alta competencia pragmática."
    
    calificacion_pragmatica_dsm5 = round(sum(puntuaciones_dsm5) / 4, 2)
    puntuaciones_totales = puntuaciones_dsm5 + puntuaciones_discurso
    calificacion_pragmatica_ampliada = round(sum(puntuaciones_totales) / 6, 2)
    
    return {
        "calificacion_pragmatica_dsm5": calificacion_pragmatica_dsm5,
        "calificacion_pragmatica_ampliada": calificacion_pragmatica_ampliada,
        "comentario_llm": comment,
        "puntuacion_a1_uso_social": puntuaciones_dsm5[0],
        "puntuacion_a2_ajuste_contexto": puntuaciones_dsm5[1],
        "puntuacion_a3_normas_conversacionales": puntuaciones_dsm5[2],
        "puntuacion_a4_comprension_no_literal": puntuaciones_dsm5[3],
        "puntuacion_a5_coherencia": puntuaciones_discurso[0],
        "puntuacion_a6_cohesion": puntuaciones_discurso[1],
        "analisis_complejidad_sintactica": 1, 
        "analisis_disfluencias": 2         
    }

def run_llama_analysis(transcription: str, pregunta_id: str) -> dict:
    """Simulación o análisis real con LLama."""
    # Como LLAMA_MODEL es None, solo llama a la simulación.
    return simulate_llama_analysis(transcription)

def save_to_database(analysis_data: dict, transcription: str, document_id: str):
    """Inserta una fila de datos de análisis en la tabla de PostgreSQL."""
    if not DB_ENGINE:
        print("⚠️ No se pudo guardar en la DB: Motor no inicializado.")
        return

    data = {
        'document_id': document_id,
        'timestamp': datetime.now().isoformat(),
        'pregunta_id': analysis_data.get('pregunta_id', 'N/A'), # Usamos .get para seguridad
        'calificacion_pragmatica_dsm5': analysis_data['calificacion_pragmatica_dsm5'],
        # [Se omiten los campos para mantener la brevedad, son idénticos a los tuyos]
        # ...
        'transcripcion_completa': transcription
    }

    # TU CÓDIGO DE INSERCIÓN SQL SE MANTIENE
    try:
        columns = ', '.join(data.keys())
        values_placeholders = ', '.join([f":{k}" for k in data.keys()])
        
        sql_insert = text(f"""
            INSERT INTO cometik_analisis ({columns})
            VALUES ({values_placeholders})
        """)
        
        with DB_ENGINE.connect() as connection:
            # Solo pasamos las claves que están en el SQL
            connection.execute(sql_insert, {k: data.get(k) for k in data.keys()})
            connection.commit()
        print(f"✅ Datos de la Pregunta {data['pregunta_id']} insertados en PostgreSQL.")
    except Exception as e:
        print(f"❌ Error al insertar datos en PostgreSQL: {e}")

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
    Recibe un archivo de audio, lo guarda (temporalmente), lo transcribe, y añade el resumen.
    """
    
    if not WHISPER_MODEL:
        raise HTTPException(status_code=503, detail="El modelo de transcripción Whisper no está cargado o falló al iniciar.")

    document_folder = STORAGE_DIR / document_id
    if not document_folder.exists():
        document_folder.mkdir()
        
    # Usamos tempfile para guardar el audio de forma segura en el disco antes de Whisper
    temp_dir = Path(tempfile.gettempdir())
    file_path = temp_dir / f"{uuid.uuid4()}_{audio_file.filename}"
    
    # 2. Guardar el archivo de audio
    try:
        with open(file_path, "wb") as buffer:
            while chunk := await audio_file.read(1024 * 1024):
                buffer.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {e}")

    # 3. Transcribir el audio usando Whisper
    try:
        # Forzar la ruta absoluta con str() para whisper, que prefiere rutas de archivo
        result = WHISPER_MODEL.transcribe(str(file_path)) 
        transcription = result["text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la transcripción de Whisper: {e}")
    finally:
        # Importante: Eliminar el archivo temporal inmediatamente después de usarlo
        os.remove(file_path) 

    # 4. Guardar la transcripción en el archivo resumen (.txt) en la carpeta de datos
    try:
        resumen_path = document_folder / f"RESUMEN_TRANSCRIPCIONES_{document_id}.txt"
        with open(resumen_path, 'a', encoding='utf-8') as f:
            f.write(f"--- PREGUNTA {pregunta_id} ---\n")
            f.write(f"Transcripción: {transcription}\n\n")
            
    except Exception as e:
        print(f"Advertencia: No se pudo escribir en el archivo resumen (.txt). {e}")
        
    # 5. Respuesta final
    return JSONResponse(content={
        "document_id": document_id,
        "pregunta_id": pregunta_id,
        "transcription": transcription,
        "message": "Audio guardado, transcrito y añadido al resumen del participante."
    })

@app.post("/analyze_text/", response_model=AnalysisResponse, tags=["Análisis LLM"])
def analyze_text(request: AnalysisRequest):
    """
    Recibe una transcripción de texto y realiza un análisis SIMULADO, 
    guardando el resultado en PostgreSQL y un archivo JSON.
    """
    
    analysis_data = run_llama_analysis(request.transcription, request.pregunta_id)
    
    # 🌟 CORRECCIÓN CLAVE: Asegurar que pregunta_id esté en los datos de análisis para la DB
    analysis_data['pregunta_id'] = request.pregunta_id
    
    document_folder = STORAGE_DIR / request.document_id
    if not document_folder.exists():
        document_folder.mkdir()
            
    # --- 2. GUARDAR EL ANÁLISIS EN LA BASE DE DATOS RELACIONAL ---
    save_to_database(analysis_data, request.transcription, request.document_id)
        
    # --- 3. Guardar el análisis completo en un archivo JSON individual ---
    try:
        analysis_filename = f"{request.pregunta_id}_analysis_{str(uuid.uuid4()).split('-')[0]}.json"
        analysis_path = document_folder / analysis_filename
        
        full_data = {
            "document_id": request.document_id,
            "pregunta_id": request.pregunta_id,
            "transcription": request.transcription,
            "analysis": analysis_data
        }
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"Advertencia: No se pudo guardar el archivo JSON de análisis. {e}")
            
    # 4. Respuesta
    return AnalysisResponse(**analysis_data)


