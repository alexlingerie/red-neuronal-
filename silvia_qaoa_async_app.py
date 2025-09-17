"""
Silvia 9.0 — Unificado con Qiskit real + QAOA asíncrono.
Mejoras: Integración de múltiples optimizadores (COBYLA, SPSA)
y mitigación de ruido para una optimización cuántica más avanzada y flexible.

Guarda como: silvia_qaoa_async_app.py
Ejecuta: python silvia_qaoa_async_app.py
Requisitos pip (recomendado):
  pip install fastapi uvicorn python-dotenv requests qiskit qiskit-aer qiskit-ibm-runtime qiskit-optimization transformers torch
"""

import os
import json
import random
import time
import uuid
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

# -----------------------------------------------------------------------------
# 1. Configuración y Persistencia
# -----------------------------------------------------------------------------
# Cargar variables de entorno
load_dotenv()

# Rutas y persistencia
BASE = Path(__file__).parent
STORAGE_PATH = BASE / "storage.json"
if not STORAGE_PATH.exists():
    STORAGE_PATH.write_text(json.dumps({
        "conocimientos": {},
        "sesgo_cuantico": {},
        "emocion": 0.0,
        "qaoa_jobs": {}
    }, indent=2, ensure_ascii=False))

def load_storage():
    with open(STORAGE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_storage(data):
    with open(STORAGE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

storage_lock = threading.Lock()

# Configs
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
IBM_QUANTUM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN")

# -----------------------------------------------------------------------------
# 2. Inicialización de Componentes Opcionales (NLP y Qiskit)
# -----------------------------------------------------------------------------
SENTIMENT_AVAILABLE = False
sentiment = None
try:
    from transformers import pipeline
    try:
        sentiment = pipeline("sentiment-analysis")
        SENTIMENT_AVAILABLE = True
        print("[Init] Sentiment pipeline cargado.")
    except Exception as e:
        print(f"[Init] No se pudo cargar pipeline de transformers, continuar sin él: {e}")
except ImportError:
    print("[Init] transformers no instalado, continuar sin él.")

QISKIT_AVAILABLE = False
USE_IBMQ = False
provider = None
try:
    from qiskit import Aer
    from qiskit_optimization import QuadraticProgram
    from qiskit.providers.aer import AerSimulator
    from qiskit.algorithms.optimizers import COBYLA, SPSA
    QISKIT_AVAILABLE = True
    print("[Init] Qiskit importado.")
    if IBM_QUANTUM_TOKEN:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
            service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_QUANTUM_TOKEN)
            provider = service
            USE_IBMQ = True
            print("[Init] QiskitRuntimeService inicializado (IBM Quantum).")
        except Exception as e:
            print(f"[Init] No se pudo inicializar QiskitRuntimeService, se usará AerSimulator fallback: {e}")
            provider = AerSimulator()
    else:
        provider = AerSimulator()
        print("[Init] No se encontró IBM_QUANTUM_TOKEN. Usando AerSimulator local.")
except ImportError as e:
    print(f"[Init] Qiskit no disponible en el entorno. La parte cuántica no funcionará: {e}")
    QISKIT_AVAILABLE = False
    provider = None
except Exception as e:
    print(f"[Init] Ocurrió un error inesperado al importar Qiskit: {e}")
    QISKIT_AVAILABLE = False
    provider = None

# ThreadPool para ejecutar trabajos QAOA en background
executor = ThreadPoolExecutor(max_workers=2)

# -----------------------------------------------------------------------------
# 3. Componentes de Silvia
# -----------------------------------------------------------------------------
class Memoria:
    def __init__(self):
        self.data = load_storage()
        self.conocimientos = self.data.get("conocimientos", {})
        self.sesgo = self.data.get("sesgo_cuantico", {})
    def guardar(self):
        with storage_lock:
            s = load_storage()
            s["conocimientos"] = self.conocimientos
            s["sesgo_cuantico"] = self.sesgo
            save_storage(s)
    def aprender(self, tema, info):
        self.conocimientos[tema] = info
        if SENTIMENT_AVAILABLE and sentiment:
            try:
                res = sentiment(info)[0]
                label = res["label"].upper()
                score = float(res.get("score", 0.5))
                key = tema.lower()
                self.sesgo.setdefault(key, 0.0)
                if label.startswith("POS"):
                    self.sesgo[key] += 0
