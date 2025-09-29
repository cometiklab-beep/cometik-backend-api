import os
import uuid
import json
from datetime import datetime
from pathlib import Path
import tempfile 
import re 
import csv 

# Importaciones de librerías CRÍTICAS (Asegúrate de que estas 4 líneas existan)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel 
from typing import Optional # Aunque no se usa directamente, es bueno tenerla si se necesitan campos opcionales.

# Intenta importar los módulos clave.
try:
    import whisper
    from sqlalchemy import create_engine, text 
except ImportError as e:
    print(f"❌ ADVERTENCIA CRÍTICA: Faltan módulos clave (Whisper/SQLAlchemy). Las funciones fallarán. {e}")
    whisper = None 
    create_engine = None 

# --- CONFIGURACIÓN DE LA API ---

API_DESCRIPTION = (
    "**API Backend Clínico y Lingüístico para la Batería COMETI-K**\n\n"
    "Esta plataforma se enfoca en realizar la **validez convergente entre el CCC-2 y COMETI-K**, "
    "una prueba psicométrica gamificada (batería cognitivo-lingüística 2D en Unity) diseñada para "
    "población de **6 a 12 años**.\n\n"
    "**Criterios de Evaluación (DSM-5 Trastorno de la Comunicación Social - Pragmático):**\n"
    "1. Uso del lenguaje en contextos sociales.\n"
    "2. Cambio del lenguaje según contexto o interlocutor.\n"
    "3. Seguimiento de normas conversacionales y narrativas.\n"
    "4. Comprensión de inferencias, metáforas y lenguaje no literal.\n\n"
    "**Objetivo:** Interpretar los resultados desde un **enfoque pragmático-lingüístico** más allá del enfoque clínico tradicional."
)

app = FastAPI(
    title="COMETI-K Backend Clínico y Lingüístico",
    description=API_DESCRIPTION,
    version="2.1.0"
)

# Directorio de almacenamiento de datos clínicos
STORAGE_DIR = Path("/opt/render/project/src/datos_clinicos") 
if not STORAGE_DIR.exists():
    STORAGE_DIR.mkdir()

# --- CARGA DE MODELOS Y BD (Global) ---

# GESTIÓN DE MEMORIA (Whisper) - Solución al error OOM
WHISPER_MODEL = None
WHISPER_DISABLED = os.environ.get('WHISPER_DISABLED', '0') 

if WHISPER_DISABLED == '1':
    print("⚠️ Modelo Whisper deshabilitado por WHISPER_DISABLED=1. Evitando OOM en Render Free.")
else:
    try:
        if whisper:
            WHISPER_MODEL = whisper.load_model("tiny") 
            print("✅ Modelo Whisper 'tiny' cargado correctamente.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo Whisper: {e}. La transcripción no estará disponible.")
        WHISPER_MODEL = None

# Configuración de la Base de Datos - AISLAMIENTO DE FALLO Y CONEXIÓN ROBUSTA
DATABASE_URL = os.environ.get('DATABASE_URL')
DB_ENGINE = None
if DATABASE_URL and create_engine:
    try:
        DB_ENGINE = create_engine(DATABASE_URL)
        with DB_ENGINE.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("✅ Motor de PostgreSQL configurado y probado correctamente.")
    except Exception as e:
        print(f"❌ ADVERTENCIA: Error crítico de conexión/URL de PostgreSQL: {e}. El servidor CONTINUARÁ. El guardado en BD no estará disponible.")
        DB_ENGINE = None 

# LLAMA SIMULACIÓN
LLAMA_MODEL = None
print("⚠️ Modo de Análisis de LLama forzado a SIMULACIÓN.")

# --- MODELOS DE DATOS (Pydantic) ---

# Modelo para los datos de registro (nuevo)
class ParticipanteRegistro(BaseModel):
    document_id: str # Documento de Identidad
    nombre: str
    genero: str
    edad: int
    acudiente_relacion: str
    acudiente_nombre: str
    direccion: str
    celular: str
    correo: str

# Modelo para la solicitud de análisis
class AnalysisRequest(BaseModel):
    document_id: str
    pregunta_id: str
    transcription: str

# Modelo para la respuesta de análisis
class AnalysisResponse(BaseModel):
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

# --- FUNCIONES DE SIMULACIÓN Y BD ---

def simulate_llama_analysis(transcription: str) -> dict:
    """Función de análisis SIMULADO."""
    word_count = len(transcription.split())
    
    if word_count < 5:
        puntuaciones_dsm5 = [0, 0, 1, 0] 
        puntuaciones_discurso = [0, 1]  
        comment = "SIMULADO: Respuesta muy corta. Baja evidencia de competencia pragmática."
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
    """Simulación del análisis LLM."""
    return simulate_llama_analysis(transcription)

def save_to_database(analysis_data: dict, transcription: str, document_id: str):
    """Inserta una fila de datos de análisis en la tabla de PostgreSQL."""
    if not DB_ENGINE:
        print("⚠️ No se pudo guardar en la DB: Motor no inicializado.")
        return
    
    data = {
        'document_id': document_id,
        'timestamp': datetime.now().isoformat(),
        'pregunta_id': analysis_data.get('pregunta_id', 'N/A'),
        'calificacion_pragmatica_dsm5': analysis_data['calificacion_pragmatica_dsm5'],
        'calificacion_pragmatica_ampliada': analysis_data['calificacion_pragmatica_ampliada'],
        'comentario_llm': analysis_data['comentario_llm'],
        'puntuacion_a1_uso_social': analysis_data['puntuacion_a1_uso_social'],
        'puntuacion_a2_ajuste_contexto': analysis_data['puntuacion_a2_ajuste_contexto'],
        'puntuacion_a3_normas_conversacionales': analysis_data['puntuacion_a3_normas_conversacionales'],
        'puntuacion_a4_comprension_no_literal': analysis_data['puntuacion_a4_comprension_no_literal'],
        'puntuacion_a5_coherencia': analysis_data['puntuacion_a5_coherencia'],
        'puntuacion_a6_cohesion': analysis_data['puntuacion_a6_cohesion'],
        'analisis_complejidad_sintactica': analysis_data['analisis_complejidad_sintactica'],
        'analisis_disfluencias': analysis_data['analisis_disfluencias'],
        'transcripcion_completa': transcription
    }

    try:
        columns = ', '.join(data.keys())
        values_placeholders = ', '.join([f":{k}" for k in data.keys()])
        sql_insert = text(f"""
            INSERT INTO cometik_analisis ({columns})
            VALUES ({values_placeholders})
        """)
        
        with DB_ENGINE.connect() as connection:
            connection.execute(sql_insert, data)
            connection.commit()
        print(f"✅ Datos de la Pregunta {data['pregunta_id']} insertados en PostgreSQL.")
    except Exception as e:
        print(f"❌ Error al insertar datos en PostgreSQL: {e}")

# --- ENDPOINTS ---

@app.get("/", tags=["Estado"])
def read_root():
    """Verifica si la API está en funcionamiento."""
    return {"status": "ok", "message": "API de COMETI-K activa y lista."}

# NUEVO ENDPOINT PARA REGISTRAR PARTICIPANTES
@app.post("/register_participant/", tags=["Registro"])
def register_participant(data: ParticipanteRegistro):
    """Guarda los datos personales del participante en la DB (reemplaza al CSV local)."""
    if not DB_ENGINE:
        raise HTTPException(status_code=503, detail="Servicio de base de datos no disponible.")
    
    try:
        sql_insert = text("""
            INSERT INTO registro_participantes (
                document_id, nombre, genero, edad, acudiente_relacion, 
                acudiente_nombre, direccion, celular, correo
            )
            VALUES (
                :document_id, :nombre, :genero, :edad, :acudiente_relacion, 
                :acudiente_nombre, :direccion, :celular, :correo
            )
            ON CONFLICT (document_id) DO NOTHING;
        """)
        
        with DB_ENGINE.connect() as connection:
            connection.execute(sql_insert, data.model_dump()) 
            connection.commit()
        return {"status": "ok", "message": f"Participante {data.document_id} registrado con éxito."}
    except Exception as e:
        print(f"❌ Error al registrar participante: {e}")
        raise HTTPException(status_code=500, detail="Error interno al guardar los datos.")


@app.post("/upload_audio/", tags=["Audio y Transcripción"])
async def upload_audio(
    document_id: str,
    pregunta_id: str,
    audio_file: UploadFile = File(...)
):
    """
    Recibe un archivo de audio. Lanza 503 si Whisper está deshabilitado.
    """
    if not WHISPER_MODEL:
        # El modelo está deshabilitado para evitar OOM
        raise HTTPException(
            status_code=503, 
            detail="El servicio de transcripción está deshabilitado. Límite de memoria de hosting alcanzado."
        )

    # Lógica de guardado de audio y transcripción (solo si Whisper estuviera activo)
    document_folder = STORAGE_DIR / document_id
    if not document_folder.exists():
        document_folder.mkdir()
    
    return JSONResponse(content={
        "document_id": document_id,
        "pregunta_id": pregunta_id,
        "transcription": "TRANSCRIPCIÓN EN PROGRESO (o deshabilitada)",
        "message": "Funcionalidad de audio activa."
    })


@app.post("/analyze_text/", response_model=AnalysisResponse, tags=["Análisis LLM"])
def analyze_text(request: AnalysisRequest):
    """
    Recibe texto y realiza un análisis SIMULADO, guardando el resultado en BD.
    """
    analysis_data = run_llama_analysis(request.transcription, request.pregunta_id)
    analysis_data['pregunta_id'] = request.pregunta_id
    
    document_folder = STORAGE_DIR / request.document_id
    if not document_folder.exists():
        document_folder.mkdir()
            
    # Guardado en DB 
    save_to_database(analysis_data, request.transcription, request.document_id)
        
    return AnalysisResponse(**analysis_data)
    
