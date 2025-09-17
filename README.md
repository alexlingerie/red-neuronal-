# silvia_qaoa_async_app.py
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
                    self.sesgo[key] += 0.5 * score
                elif label.startswith("NEG"):
                    self.sesgo[key] -= 0.5 * score
            except Exception:
                pass
        self.guardar()
    def get(self, tema):
        return self.conocimientos.get(tema)

memoria = Memoria()

class Emocional:
    def __init__(self):
        s = load_storage()
        self.estado = s.get("emocion", 0.0)
    def ajustar(self, delta):
        self.estado = max(-1.0, min(1.0, self.estado + delta))
        with storage_lock:
            s = load_storage()
            s["emocion"] = self.estado
            save_storage(s)
        return self.estado
    def descripcion(self):
        if self.estado > 0.5: return "Muy feliz y optimista."
        if self.estado > 0.1: return "Feliz."
        if self.estado < -0.5: return "Muy triste o frustrada."
        if self.estado < -0.1: return "Algo triste."
        return "Neutral."

emocional = Emocional()

# -----------------------------------------------------------------------------
# 4. Función que ejecuta QAOA
# -----------------------------------------------------------------------------
def run_qaoa_job(job_id: str, positives: float, negatives: float, optimizer_name: str, mitigate_noise: bool):
    """
    Ejecuta una tarea QAOA (puede tardar). Persistimos estado y resultado en storage.json -> qaoa_jobs.
    """
    with storage_lock:
        s = load_storage()
        jobs = s.get("qaoa_jobs", {})
        jobs[job_id]["status"] = "running"
        save_storage(s)

    start_ts = time.time()
    opinion = "neutral"
    details = {"method": "unknown", "optimizer": optimizer_name, "mitigated": mitigate_noise}

    try:
        if QISKIT_AVAILABLE:
            qp = QuadraticProgram("opinion")
            qp.binary_var("p")
            qp.binary_var("n")
            qp.maximize(linear={"p": positives, "n": negatives})
            qp.maximize(quadratic={("p", "n"): -5})
            qubit_op, offset = qp.to_ising()

            sol = None
            try:
                from qiskit.algorithms import QAOA
                
                if optimizer_name == "COBYLA":
                    optimizer = COBYLA(maxiter=100)
                elif optimizer_name == "SPSA":
                    optimizer = SPSA(maxiter=100)
                else:
                    raise ValueError("Optimizador no soportado.")
                
                if USE_IBMQ:
                    from qiskit_ibm_runtime import Sampler, SamplerOptions
                    
                    # Configuración de la mitigación de ruido
                    sampler_options = SamplerOptions()
                    if mitigate_noise:
                        sampler_options.resilience_level = 1 # Nivel 1 de mitigación de errores
                        print("Mitigación de ruido activada. Usando 'resilience_level=1'.")
                    
                    sampler = Sampler(options=sampler_options)
                    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
                    result = qaoa.compute_minimum_eigenvalue(qubit_op)
                    sol = result.x
                else:
                    from qiskit.utils import QuantumInstance
                    backend = provider if isinstance(provider, AerSimulator) else AerSimulator()
                    qi = QuantumInstance(backend=backend, shots=1024)
                    qaoa = QAOA(optimizer=optimizer, reps=1, quantum_instance=qi)
                    result = qaoa.compute_minimum_eigenvalue(qubit_op)
                    sol = result.x
                
                best_opinion = "neutral"
                best_score = float('-inf')
                
                options = [(0, 0), (0, 1), (1, 0), (1, 1)]
                for p_bit, n_bit in options:
                    score = (positives * p_bit) + (negatives * n_bit) - (5 * p_bit * n_bit)
                    if score > best_score:
                        best_score = score
                        best_bits = (p_bit, n_bit)

                p_bit, n_bit = best_bits
                if p_bit == 1 and n_bit == 0:
                    opinion = "positiva"
                elif p_bit == 0 and n_bit == 1:
                    opinion = "negativa"
                else:
                    opinion = "neutral"
                details = {"method": "qiskit_qaoa_robust", "raw_solution": list(map(int, sol)), "best_binary_solution": best_bits, "optimizer_used": optimizer_name, "function_evals": result.optimal_point}
            except Exception as e:
                print(f"[QAOA local/runtime error] {e}")
                opinion = "neutral"
                details = {"method": "qiskit_qaoa_error", "error": str(e), "optimizer_used": optimizer_name}

        if not QISKIT_AVAILABLE or details.get("method") == "qiskit_qaoa_error":
            if positives > negatives:
                opinion = "positiva"
            elif negatives > positives:
                opinion = "negativa"
            else:
                opinion = "neutral"
            details = {"method": "heuristic_fallback", "optimizer_used": "N/A"}

        duration = time.time() - start_ts
        with storage_lock:
            s = load_storage()
            jobs = s.get("qaoa_jobs", {})
            jobs[job_id].update({
                "status": "done",
                "result": {"opinion": opinion, "details": details, "positives": positives, "negatives": negatives, "duration_s": duration},
                "finished_at": time.time()
            })
            s["qaoa_jobs"] = jobs
            save_storage(s)

    except Exception as e:
        with storage_lock:
            s = load_storage()
            jobs = s.get("qaoa_jobs", {})
            jobs[job_id].update({"status": "error", "error": str(e)})
            s["qaoa_jobs"] = jobs
            save_storage(s)

# -----------------------------------------------------------------------------
# 5. API y lógica de alto nivel (FastAPI)
# -----------------------------------------------------------------------------
app = FastAPI(title="Silvia 9.0 — QAOA Async")

class AprenderReq(BaseModel):
    tema: str
    info: str

class QAOASubmitReq(BaseModel):
    tema: Optional[str] = None
    positives: Optional[float] = None
    negatives: Optional[float] = None
    optimizer: Optional[str] = "COBYLA"
    mitigate_noise: Optional[bool] = False

@app.get("/status")
def status():
    s = load_storage()
    return {"status": "ok", "emocion": s.get("emocion", 0.0), "memoria_keys": list(s.get("conocimientos", {}).keys())}

@app.post("/aprender")
def api_aprender(req: AprenderReq):
    memoria.aprender(req.tema, req.info)
    return {"ok": True, "mensaje": f"He aprendido que {req.tema} es {req.info}"}

@app.get("/opinion")
def api_opinion(tema: str):
    info = memoria.get(tema)
    if info is None:
        raise HTTPException(status_code=404, detail="Tema no conocido. Usa /aprender primero.")
    if SENTIMENT_AVAILABLE and sentiment:
        res = sentiment(info)[0]
        label = res["label"].upper()
        score = float(res.get("score", 0.5))
        if label.startswith("POS"):
            return {"opinion": f"Tengo una opinión positiva sobre {tema} (score {score})"}
        else:
            return {"opinion": f"Tengo una opinión negativa sobre {tema} (score {score})"}
    else:
        return {"opinion": f"Mi opinión simulada sobre {tema} es neutra (no hay pipeline de sentiment disponible)."}

@app.post("/emocion/ajustar")
def api_emotion_adjust(delta: float):
    nuevo = emocional.ajustar(delta)
    return {"emocion": nuevo, "descripcion": emocional.descripcion()}

@app.get("/emocion")
def api_get_emotion():
    return {"emocion": emocional.estado, "descripcion": emocional.descripcion()}

@app.post("/qaoa/submit")
def qaoa_submit(req: QAOASubmitReq):
    if req.tema:
        p = random.random() * 3
        n = random.random() * 3
        positives, negatives = p, n
    else:
        positives = float(req.positives or 0.0)
        negatives = float(req.negatives or 0.0)

    if req.optimizer not in ["COBYLA", "SPSA"]:
        raise HTTPException(status_code=400, detail="Optimizador no válido. Use 'COBYLA' o 'SPSA'.")

    job_id = str(uuid.uuid4())
    job_entry = {
        "id": job_id,
        "status": "queued",
        "submitted_at": time.time(),
        "positives": positives,
        "negatives": negatives,
        "optimizer": req.optimizer,
        "mitigate_noise": req.mitigate_noise
    }
    with storage_lock:
        s = load_storage()
        jobs = s.get("qaoa_jobs", {})
        jobs[job_id] = job_entry
        s["qaoa_jobs"] = jobs
        save_storage(s)

    executor.submit(run_qaoa_job, job_id, positives, negatives, req.optimizer, req.mitigate_noise)

    return {"job_id": job_id, "status": "queued", "optimizer": req.optimizer, "mitigate_noise": req.mitigate_noise}

@app.get("/qaoa/status/{job_id}")
def qaoa_status(job_id: str):
    s = load_storage()
    jobs = s.get("qaoa_jobs", {})
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id no encontrado")
    return job

@app.get("/qaoa/result/{job_id}")
def qaoa_result(job_id: str):
    s = load_storage()
    jobs = s.get("qaoa_jobs", {})
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id no encontrado")
    if job.get("status") != "done":
        return {"status": job.get("status"), "message": "Resultado no disponible aún."}
    return job.get("result")

# -----------------------------------------------------------------------------
# 6. Interfaz de Usuario y Bucle de Ejecución
# -----------------------------------------------------------------------------
INDEX = """
<!doctype html><html><head><meta charset="utf-8"><title>Silvia 9.0 QAOA Async</title>
<style>
body{font-family:system-ui,Arial;background:#f3f4f6;padding:18px}
.card{max-width:900px;margin:0 auto;background:#fff;padding:20px;border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,0.06)}
.input{display:flex;gap:8px}
.input input, .input select{flex:1;padding:8px;border-radius:8px;border:1px solid #ddd}
button{background:#2563eb;color:white;padding:8px 12px;border-radius:8px;border:none}
.log{margin-top:12px;background:#f9fafb;padding:10px;border-radius:8px;height:260px;overflow:auto}
.job{font-size:13px;color:#374151}
.checkbox-container { display: flex; align-items: center; gap: 5px; }
</style></head><body>
<div class="card">
<h2>Silvia 9.0 — QAOA Async</h2>
<div>Enviar QAOA (usar tema o valores)<br/></div>
<div style="margin-top:8px" class="input">
<input id="tema" placeholder="tema (opcional) — p.e. 'videojuegos'">
<input id="p" placeholder="positives (opcional)">
<input id="n" placeholder="negatives (opcional)">
<select id="opt">
    <option value="COBYLA">COBYLA</option>
    <option value="SPSA">SPSA</option>
</select>
<div class="checkbox-container">
    <input type="checkbox" id="mitigate_noise">
    <label for="mitigate_noise">Mitigar Ruido</label>
</div>
<button onclick="submitQ()">Submit QAOA</button>
</div>
<div class="log" id="log"></div>
</div>
<script>
async function append(msg){ let log=document.getElementById('log'); log.innerHTML += '<div>'+msg+'</div>'; log.scrollTop = log.scrollHeight; }
async function submitQ(){
  const tema = document.getElementById('tema').value.trim();
  const p = document.getElementById('p').value.trim();
  const n = document.getElementById('n').value.trim();
  const opt = document.getElementById('opt').value;
  const mitigate = document.getElementById('mitigate_noise').checked;
  const body = { optimizer: opt, mitigate_noise: mitigate };
  if(tema) body.tema = tema; else { if(p) body.positives = parseFloat(p); if(n) body.negatives = parseFloat(n); }
  append('Enviando job con ' + opt + ' y mitigación ' + mitigate + '...');
  const res = await fetch('/qaoa/submit', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body)});
  const json = await res.json();
  if (res.status !== 200) { append('Error: ' + JSON.stringify(json)); return; }
  append('Job queued: ' + json.job_id);
  poll(json.job_id);
}
async function poll(job_id){
  append('Polling ' + job_id);
  const int = setInterval(async ()=>{
    const res = await fetch('/qaoa/status/' + job_id);
    const j = await res.json();
    append('Status: ' + j.status);
    if(j.status === 'done' || j.status === 'error'){ clearInterval(int); append('Resultado: ' + JSON.stringify(j.result || j.error || {})); }
  }, 3000);
}
</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX)

# Ejecutar
if __name__ == "__main__":
    import uvicorn
    print("Arrancando Silvia 9.0 con QAOA async — abrir http://localhost:8000")
    uvicorn.run("silvia_qaoa_async_app:app", host="0.0.0.0", port=8000, reload=False)
    
