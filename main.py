import os
import uuid
import whisper
import json
import re 
import csv 
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sqlalchemy import create_engine, text 

# (A) CONFIGURACIÓN LLAMA: Define la ruta del modelo GGUF (la carga está deshabilitada)
LLAMA_MODEL_PATH = os.path.join(os.path.dirname(__file__), "llama_model.gguf")

# --- DESCRIPCIÓN EXTENSA DE LA API ---
API_DESCRIPTION = (
    "API para la recolección, transcripción y análisis automatizado de respuestas verbales infantiles "
    "en el marco de la validación convergente entre el CCC-2 y COMETI-K. Esta batería cognitivo-lingüística "
    "gamificada en Unity evalúa los criterios del DSM-5 para el trastorno de la comunicación social (pragmático), "
    "integrando métricas discursivas para un análisis lingüístico robusto."
)

# --- CONFIGURACIÓN DE LA API ---
app = FastAPI(
    title="COMETI-K Backend Clínico y Lingüístico",
    description=API_DESCRIPTION,
    version="1.0.0"
)

# Directorio donde se guardarán los archivos de audio y resultados
STORAGE_DIR = "datos_clinicos" 
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# --- CONFIGURACIÓN DE LA BASE DE DATOS ---
# Render inyectará la variable DATABASE_URL automáticamente
DATABASE_URL = os.environ.get('DATABASE_URL')
DB_ENGINE = None
if DATABASE_URL:
    try:
        # Crea el motor de conexión a la base de datos
        DB_ENGINE = create_engine(DATABASE_URL)
        print("✅ Motor de PostgreSQL configurado correctamente.")
    except Exception as e:
        print(f"❌ Error al crear el motor de PostgreSQL: {e}")

# --- MODELOS DE DATOS ---
class AnalysisRequest(BaseModel):
    """Modelo para solicitar el análisis de texto por el LLM."""
    document_id: str
    pregunta_id: str
    transcription: str

class AnalysisResponse(BaseModel):
    """Modelo de la respuesta del LLM con puntuaciones clínicas y lingüísticas (0, 1, 2)."""
    # ENFOQUE DUAL: Calificación clínica (A1-A4) y Calificación ampliada (A1-A6)
    calificacion_pragmatica_dsm5: float  
    calificacion_pragmatica_ampliada: float
    comentario_llm: str
    
    # CRITERIOS DSM-5 PUROS (A1-A4)
    puntuacion_a1_uso_social: int
    puntuacion_a2_ajuste_contexto: int
    puntuacion_a3_normas_conversacionales: int
    puntuacion_a4_comprension_no_literal: int
    
    # CRITERIOS LINGÜÍSTICOS ADICIONALES (A5-A6)
    puntuacion_a5_coherencia: int         
    puntuacion_a6_cohesion: int           
    
    # ANÁLISIS DISCURSIVOS EXTRA
    analisis_complejidad_sintactica: int
    analisis_disfluencias: int

# --- CARGA DE MODELOS ---
# Carga el modelo de Whisper
try:
    WHISPER_MODEL = whisper.load_model("base")
    print("✅ Modelo Whisper 'base' cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo Whisper: {e}")
    WHISPER_MODEL = None

# (B) CARGA REAL DEL MODELO LLAMA - DESHABILITADA
LLAMA_MODEL = None
print("⚠️ Modo de Análisis de LLama forzado a SIMULACIÓN para evitar Timeouts.")


# --- FUNCIONES DE SIMULACIÓN Y ANÁLISIS ---

def simulate_llama_analysis(transcription: str) -> dict:
    """Función de análisis SIMULADO, con las 8 métricas requeridas."""
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
    
    # Cálculo de promedios para simulación
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
    """
    Función de análisis con prompt estructurado para los criterios DSM-5 y lingüísticos (0, 1, 2).
    """
    if LLAMA_MODEL is None:
        return simulate_llama_analysis(transcription)

    # --- CRITERIOS DEFINITIVOS PARA EL PROMPT (Si LLAMA estuviera activo) ---
    CRITERIOS_ANALISIS = (
        "Instrucciones de puntuación (0=Ausencia del Criterio/Déficit, 1=Presencia Parcial, 2=Presencia Clara del Criterio):\n"
        "1. A1_Uso_Social (Uso del lenguaje en contextos sociales):\n"
        "2. A2_Ajuste_Contexto (Cambio del lenguaje según contexto o interlocutor):\n"
        "3. A3_Normas_Conversacionales (Seguimiento de normas conversacionales y narrativas):\n"
        "4. A4_Comprension_No_Literal (Comprensión de inferencias, metáforas y lenguaje no literal):\n\n"
        "5. A5_Coherencia (Nivel Discursivo - Lógica en el encadenamiento de ideas):\n"
        "6. A6_Cohesión (Nivel Lingüístico - Uso de conectores y referentes):\n\n"
        "7. Complejidad_Sintáctica (0=Baja, 1=Media, 2=Alta):\n"
        "8. Disfluencias (0=Alta Presencia, 1=Media, 2=Baja/Nula Presencia):\n"
    )
    
    SYSTEM_PROMPT = (
        "Eres un experto en el diagnóstico del Trastorno de la Comunicación Social (TCS) con un enfoque pragmático-lingüístico. "
        "Tu tarea es puntuar la competencia pragmática y discursiva de la respuesta de un niño/a (6-12 años) "
        "usando la escala discreta 0, 1, 2. La Calificación DSM-5 es el promedio de A1-A4. La Calificación Ampliada es el promedio de A1-A6."
    )
    
    USER_PROMPT = (
        f"PREGUNTA ID: {pregunta_id}\n"
        f"TEXTO A ANALIZAR: '{transcription}'\n\n"
        f"{CRITERIOS_ANALISIS}"
        "Genera un comentario conciso (máx 40 palabras) y la Puntuación Pragmática Global final."
        "IMPORTANTE: Proporciona la salida en el siguiente formato EXACTO:\n"
        "Puntuaciones: A1_Uso_Social: [X], A2_Ajuste_Contexto: [X], A3_Normas_Conversacionales: [X], A4_Comprension_No_Literal: [X], A5_Coherencia: [X], A6_Cohesión: [X], Complejidad_Sintáctica: [X], Disfluencias: [X]\n"
        "Comentario: [Texto conciso]\n"
        "DSM5: [Puntuación Decimal XX]\n"
        "AMPLIADA: [Puntuación Decimal XX]\n"
    )

    prompt = f"### System: {SYSTEM_PROMPT}\n### User: {USER_PROMPT}\n### Assistant:"

    try:
        # Si LLAMA_MODEL está activo, se haría la llamada real aquí

        # --- LÓGICA DE PARSEO Y CÁLCULO ---
        # (El código de parsing se mantiene, asumiendo una respuesta estructurada o usando el fallback de simulación)
        
        # Simulando una respuesta estructurada para fines de prueba/parsing si LLAMA_MODEL es None
        response_text = "Puntuaciones: A1_Uso_Social: [2], A2_Ajuste_Contexto: [2], A3_Normas_Conversacionales: [1], A4_Comprension_No_Literal: [0], A5_Coherencia: [2], A6_Cohesión: [1], Complejidad_Sintáctica: [2], Disfluencias: [2]\nComentario: Placeholder para prueba de parsing.\nDSM5: 1.25\nAMPLIADA: 1.33"

        puntuaciones_match = re.search(r"Puntuaciones:\s*(.*)", response_text)
        dsm5_score_match = re.search(r"DSM5:\s*(\d\.\d+)", response_text)
        ampliada_score_match = re.search(r"AMPLIADA:\s*(\d\.\d+)", response_text)
        comment_match = re.search(r"Comentario:\s*(.*)", response_text)

        # Extracción de puntuaciones discretas
        scores_dict = {}
        if puntuaciones_match:
            scores_str = puntuaciones_match.group(1).replace('[', '').replace(']', '')
            criterios = re.findall(r"([A-Za-z0-9_]+):\s*(\d)", scores_str)
            for key, value in criterios:
                scores_dict[key] = int(value)

        # Cálculo de promedios
        score_keys_dsm5 = ['A1_Uso_Social', 'A2_Ajuste_Contexto', 'A3_Normas_Conversacionales', 'A4_Comprension_No_Literal']
        score_keys_ampliada = score_keys_dsm5 + ['A5_Coherencia', 'A6_Cohesión']
        
        # Fallback de puntuaciones
        dsm5_scores = [scores_dict.get(k, 0) for k in score_keys_dsm5]
        ampliada_scores = [scores_dict.get(k, 0) for k in score_keys_ampliada]

        # Puntuación Global (DSM-5 Puro)
        calificacion_pragmatica_dsm5 = float(dsm5_score_match.group(1)) if dsm5_score_match else (sum(dsm5_scores) / len(dsm5_scores))
        
        # Puntuación Ampliada (Lingüístico/Discursivo)
        calificacion_pragmatica_ampliada = float(ampliada_score_match.group(1)) if ampliada_score_match else (sum(ampliada_scores) / len(ampliada_scores))

        comentario_llm = comment_match.group(1).strip() if comment_match else response_text[:100]

        return {
            "calificacion_pragmatica_dsm5": round(calificacion_pragmatica_dsm5, 2),
            "calificacion_pragmatica_ampliada": round(calificacion_pragmatica_ampliada, 2),
            "comentario_llm": comentario_llm,
            "puntuacion_a1_uso_social": scores_dict.get('A1_Uso_Social', 0),
            "puntuacion_a2_ajuste_contexto": scores_dict.get('A2_Ajuste_Contexto', 0),
            "puntuacion_a3_normas_conversacionales": scores_dict.get('A3_Normas_Conversacionales', 0),
            "puntuacion_a4_comprension_no_literal": scores_dict.get('A4_Comprension_No_Literal', 0),
            "puntuacion_a5_coherencia": scores_dict.get('A5_Coherencia', 0),
            "puntuacion_a6_cohesion": scores_dict.get('A6_Cohesión', 0),
            "analisis_complejidad_sintactica": scores_dict.get('Complejidad_Sintáctica', 1),
            "analisis_disfluencias": scores_dict.get('Disfluencias', 1)
        }
    except Exception as e:
        print(f"❌ Error crítico en el LLama o Parseo: {e}. Volviendo a la simulación.")
        return simulate_llama_analysis(transcription)

def save_to_database(analysis_data: dict, transcription: str, document_id: str):
    """Inserta una fila de datos de análisis en la tabla de PostgreSQL."""
    if not DB_ENGINE:
        print("⚠️ No se pudo guardar en la DB: Motor no inicializado.")
        return

    # Los datos se mapean a las columnas exactas de la tabla SQL
    data = {
        'document_id': document_id,
        'timestamp': datetime.now().isoformat(),
        # 'pregunta_id' ya viene en analysis_data gracias al fix en analyze_text
        'pregunta_id': analysis_data['pregunta_id'], 
        'calificacion_pragmatica_dsm5': analysis_data['calificacion_pragmatica_dsm5'],
        'calificacion_pragmatica_ampliada': analysis_data['calificacion_pragmatica_ampliada'],
        'puntuacion_a1_uso_social': analysis_data['puntuacion_a1_uso_social'],
        'puntuacion_a2_ajuste_contexto': analysis_data['puntuacion_a2_ajuste_contexto'],
        'puntuacion_a3_normas_conversacionales': analysis_data['puntuacion_a3_normas_conversacionales'],
        'puntuacion_a4_comprension_no_literal': analysis_data['puntuacion_a4_comprension_no_literal'],
        'puntuacion_a5_coherencia': analysis_data['puntuacion_a5_coherencia'],
        'puntuacion_a6_cohesion': analysis_data['puntuacion_a6_cohesion'],
        'analisis_complejidad_sintactica': analysis_data['analisis_complejidad_sintactica'],
        'analisis_disfluencias': analysis_data['analisis_disfluencias'],
        'comentario_llm': analysis_data['comentario_llm'],
        'transcripcion_completa': transcription
    }

    # Se construyen dinámicamente las columnas y los placeholders para la inserción
    columns = ', '.join(data.keys())
    values_placeholders = ', '.join([f":{k}" for k in data.keys()])
    
    sql_insert = text(f"""
        INSERT INTO cometik_analisis ({columns})
        VALUES ({values_placeholders})
    """)
    
    try:
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

@app.post("/upload_audio/", tags=["Audio y Transcripción"])
async def upload_audio(
    document_id: str,
    pregunta_id: str,
    audio_file: UploadFile = File(...)
):
    """
    Recibe un archivo de audio, lo guarda, lo transcribe usando Whisper, 
    y añade la transcripción al archivo resumen (.txt) del participante.
    """
    
    if not WHISPER_MODEL:
        raise HTTPException(status_code=503, detail="El modelo de transcripción Whisper no está cargado.")

    # 1. Definición de rutas y creación de la carpeta del participante
    file_extension = os.path.splitext(audio_file.filename)[1]
    document_folder = os.path.join(STORAGE_DIR, document_id)
    if not os.path.exists(document_folder):
        os.makedirs(document_folder)
        
    file_name = f"{pregunta_id}_{str(uuid.uuid4()).split('-')[0]}{file_extension}"
    file_path = os.path.join(document_folder, file_name)

    # 2. Guardar el archivo de audio
    try:
        with open(file_path, "wb") as buffer:
            while chunk := await audio_file.read(1024 * 1024):
                buffer.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {e}")

    # 3. Transcribir el audio usando Whisper
    try:
        result = WHISPER_MODEL.transcribe(file_path)
        transcription = result["text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la transcripción de Whisper: {e}")

    # 4. Guardar la transcripción en el archivo resumen (.txt)
    try:
        resumen_filename = f"RESUMEN_TRANSCRIPCIONES_{document_id}.txt"
        resumen_path = os.path.join(document_folder, resumen_filename)
        
        with open(resumen_path, 'a', encoding='utf-8') as f:
            f.write(f"--- PREGUNTA {pregunta_id} ---\n")
            f.write(f"Audio: {file_name}\n") 
            f.write(f"Transcripción: {transcription}\n\n")
            
    except Exception as e:
        print(f"Advertencia: No se pudo escribir en el archivo resumen (.txt). {e}")
        
    # 5. Respuesta final
    return JSONResponse(content={
        "document_id": document_id,
        "pregunta_id": pregunta_id,
        "audio_path": file_path,
        "transcription": transcription,
        "message": "Audio guardado, transcrito y añadido al resumen del participante."
    })

@app.post("/analyze_text/", response_model=AnalysisResponse, tags=["Análisis LLM"])
def analyze_text(request: AnalysisRequest):
    """
    Recibe una transcripción de texto y realiza un análisis SIMULADO (o real), 
    guardando el resultado en un archivo CSV y en la base de datos PostgreSQL.
    """
    
    # 1. Realizar el análisis (simulación o real)
    analysis_data = run_llama_analysis(request.transcription, request.pregunta_id)
    
    # 🌟 CORRECCIÓN CLAVE: Agregamos pregunta_id al diccionario de datos analizados.
    # Esto soluciona el KeyError al llamar a save_to_database.
    analysis_data['pregunta_id'] = request.pregunta_id
    
    document_folder = os.path.join(STORAGE_DIR, request.document_id)
    if not os.path.exists(document_folder):
        os.makedirs(document_folder)
            
    # --- 2. GUARDAR EL ANÁLISIS EN UN ARCHIVO CSV CONSOLIDADO (MÉTODO LOCAL) ---
    try:
        csv_filename = f"ANALISIS_RESUMEN_{request.document_id}.csv"
        csv_path = os.path.join(document_folder, csv_filename)
        
        # Define los campos del archivo CSV
        fieldnames = [
            'timestamp', 
            'pregunta_id', 
            'calificacion_pragmatica_dsm5',       
            'calificacion_pragmatica_ampliada',   
            'puntuacion_a1_uso_social', 
            'puntuacion_a2_ajuste_contexto', 
            'puntuacion_a3_normas_conversacionales', 
            'puntuacion_a4_comprension_no_literal',  
            'puntuacion_a5_coherencia', 
            'puntuacion_a6_cohesion', 
            'analisis_complejidad_sintactica', 
            'analisis_disfluencias', 
            'comentario_llm', 
            'transcripcion_completa'
        ]
        
        # Datos a escribir en la fila
        csv_data = {
            'timestamp': datetime.now().isoformat(),
            'pregunta_id': analysis_data['pregunta_id'],
            'calificacion_pragmatica_dsm5': analysis_data['calificacion_pragmatica_dsm5'],
            'calificacion_pragmatica_ampliada': analysis_data['calificacion_pragmatica_ampliada'],
            'puntuacion_a1_uso_social': analysis_data['puntuacion_a1_uso_social'],
            'puntuacion_a2_ajuste_contexto': analysis_data['puntuacion_a2_ajuste_contexto'],
            'puntuacion_a3_normas_conversacionales': analysis_data['puntuacion_a3_normas_conversacionales'],
            'puntuacion_a4_comprension_no_literal': analysis_data['puntuacion_a4_comprension_no_literal'],
            'puntuacion_a5_coherencia': analysis_data['puntuacion_a5_coherencia'],
            'puntuacion_a6_cohesion': analysis_data['puntuacion_a6_cohesion'],
            'analisis_complejidad_sintactica': analysis_data['analisis_complejidad_sintactica'],
            'analisis_disfluencias': analysis_data['analisis_disfluencias'],
            'comentario_llm': analysis_data['comentario_llm'].replace('\n', ' '), 
            'transcripcion_completa': request.transcription.replace('\n', ' ')
        }

        # Comprueba si el archivo existe para escribir el encabezado
        is_new_file = not os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if is_new_file:
                writer.writeheader() # Escribe el encabezado solo si es un archivo nuevo
            
            writer.writerow(csv_data)

    except Exception as e:
        print(f"Advertencia: No se pudo escribir en el archivo CSV de análisis. {e}")
        
    # --- 3. GUARDAR EL ANÁLISIS EN LA BASE DE DATOS RELACIONAL ---
    save_to_database(analysis_data, request.transcription, request.document_id)
        
    # --- 4. Guardar el análisis completo en un archivo JSON individual ---
    try:
        analysis_filename = f"{request.pregunta_id}_analysis_{str(uuid.uuid4()).split('-')[0]}.json"
        analysis_path = os.path.join(document_folder, analysis_filename)
        
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
        
    # 5. Respuesta
    return AnalysisResponse(**analysis_data)

