#!/usr/bin/env bash
set -euo pipefail

# Prueba de aceptación funcional de la PoC.
# Requisitos en el host: curl y python3.
# Ejecutar después de: docker compose -f infrastructure/docker-compose.yml up --build

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
API_URL="${API_URL:-http://localhost:8000}"
GLM_URL="${GLM_URL:-http://localhost:8080}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:4200}"
OCR_SAMPLE="${OCR_SAMPLE:-$ROOT_DIR/datasets/ocr_samples/documento_escaneado_prueba_ocr_pagos.jpg}"
RUN_EVALUATION="${RUN_EVALUATION:-false}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
WARN=0

ok() {
  echo "[OK] $1"
  PASS=$((PASS + 1))
}

warn() {
  echo "[WARN] $1"
  WARN=$((WARN + 1))
}

fail() {
  echo "[FAIL] $1" >&2
  exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl no está instalado"
command -v python3 >/dev/null 2>&1 || fail "python3 no está instalado"
[[ -f "$OCR_SAMPLE" ]] || fail "No existe la muestra OCR: $OCR_SAMPLE"

echo "== Smoke test Adaptive RAG Payments =="
echo "API: $API_URL"
echo "GLM-OCR: $GLM_URL"
echo "Frontend: $FRONTEND_URL"
echo

# 1) vLLM debe publicar GLM-OCR.
curl -fsS --max-time 15 "$GLM_URL/v1/models" > "$TMP_DIR/models.json" \
  || fail "GLM-OCR/vLLM no responde en $GLM_URL/v1/models"
python3 - "$TMP_DIR/models.json" <<'PY' || fail "vLLM responde pero no publica el modelo glm-ocr"
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
ids = {str(x.get("id", "")) for x in p.get("data", [])}
assert "glm-ocr" in ids, ids
PY
ok "vLLM publica el modelo glm-ocr"

# 2) Servicios base del backend.
curl -fsS --max-time 15 "$API_URL/api/v1/health" > "$TMP_DIR/health.json" \
  || fail "El backend no responde en /api/v1/health"
python3 - "$TMP_DIR/health.json" <<'PY' || fail "Qdrant, Memgraph, RAGLight o GLM-OCR no están listos"
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
required = ["qdrant", "memgraph", "raglight", "glm_ocr"]
missing = [k for k in required if not p.get(k)]
assert not missing, {"missing": missing, "health": p}
PY
ok "Backend + Qdrant + Memgraph + RAGLight + GLM-OCR responden"

# Regresión v19: el bootstrap no debe monopolizar el event loop de FastAPI.
curl -fsS --max-time 15 "$API_URL/docs" > /dev/null \
  || fail "FastAPI responde en health pero /docs está bloqueado"
ok "FastAPI mantiene /docs disponible"

# La UI no debe depender de que dataset-seed termine.
curl -fsS --max-time 15 "$FRONTEND_URL/" > "$TMP_DIR/frontend.html" \
  || fail "El frontend Angular/Nginx no responde en $FRONTEND_URL"
grep -qi "<html\|<app-root" "$TMP_DIR/frontend.html" \
  || fail "El frontend respondió pero no devolvió HTML esperado"
ok "Frontend Angular está disponible sin esperar al dataset-seed"

if python3 - "$TMP_DIR/health.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if p.get("lightrag") else 1)
PY
then
  ok "LightRAG está disponible"
else
  warn "LightRAG no está disponible; es esperado si no se configuró proveedor OpenAI-compatible"
fi

# 3) OCR real: fuerza GLM-OCR sobre una imagen incluida en el proyecto.
OCR_HTTP_CODE="$(curl -sS --max-time 600 \
  -o "$TMP_DIR/ocr.json" -w "%{http_code}" \
  -X POST "$API_URL/api/v1/documents/ingest?ocr=glm" \
  -F "file=@$OCR_SAMPLE")" || fail "La ingesta OCR real no pudo conectarse al backend"
if [[ "$OCR_HTTP_CODE" -lt 200 || "$OCR_HTTP_CODE" -ge 300 ]]; then
  echo "Respuesta OCR (HTTP $OCR_HTTP_CODE):" >&2
  cat "$TMP_DIR/ocr.json" >&2 || true
  echo >&2
  fail "La ingesta OCR real falló"
fi
python3 - "$TMP_DIR/ocr.json" <<'PY' || fail "La respuesta OCR no confirma uso de GLM-OCR"
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p.get("processor") == "glm-ocr", p
assert p.get("ocr_used") is True, p
assert int(p.get("chunks", 0)) > 0, p
PY
ok "GLM-OCR procesó una imagen real y produjo chunks"

# 4) Native RAG: debe recuperar el documento de idempotencia ya precargado.
cat > "$TMP_DIR/native-request.json" <<'JSON'
{
  "question": "¿Cómo evita la idempotencia cobros duplicados cuando hay timeout?",
  "strategy": "native_rag",
  "top_k": 5,
  "include_trace": true,
  "include_evidence": true
}
JSON
curl -fsS --max-time 60 \
  -X POST "$API_URL/api/v1/query" \
  -H "Content-Type: application/json" \
  --data-binary @"$TMP_DIR/native-request.json" > "$TMP_DIR/native.json" \
  || fail "Native RAG no respondió"
python3 - "$TMP_DIR/native.json" <<'PY' || fail "Native RAG no recuperó idempotency.md"
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
sources = [str(x.get("source", "")) for x in p.get("contexts", [])]
assert p.get("executed_strategy") == "native_rag", p
assert "idempotency.md" in sources, sources
PY
ok "Native RAG recupera idempotency.md"

# 5) Adaptive Routing: la consulta relacional debe ir a GraphRAG.
curl -fsS --max-time 60 \
  -X POST "$API_URL/api/v1/query" \
  -H "Content-Type: application/json" \
  --data-binary @"$ROOT_DIR/infrastructure/requests/query-auto.json" > "$TMP_DIR/auto.json" \
  || fail "Adaptive Routing no respondió"
python3 - "$TMP_DIR/auto.json" <<'PY' || fail "Adaptive Routing no seleccionó GraphRAG"
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p.get("executed_strategy") == "graphrag", p
assert len(p.get("contexts", [])) > 0, p
PY
ok "Adaptive Routing selecciona GraphRAG para una consulta relacional"

# 6) Grafo: debe haber al menos una relación para AUTORIZACION.
curl -fsS --max-time 30 "$API_URL/api/v1/graph/neighborhood/AUTORIZACION" > "$TMP_DIR/graph.json" \
  || fail "La consulta de vecindad de Memgraph falló"
python3 - "$TMP_DIR/graph.json" <<'PY' || fail "Memgraph respondió sin relaciones para AUTORIZACION"
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(p.get("edges", [])) > 0, p
PY
ok "Memgraph devuelve relaciones para AUTORIZACION"

# 7) RAGLight real: debe recuperar al menos un contexto.
cat > "$TMP_DIR/raglight-request.json" <<'JSON'
{
  "question": "¿Qué significa el código 05 en una autorización?",
  "strategy": "raglight",
  "top_k": 5,
  "include_trace": true,
  "include_evidence": true
}
JSON
curl -fsS --max-time 120 \
  -X POST "$API_URL/api/v1/query" \
  -H "Content-Type: application/json" \
  --data-binary @"$TMP_DIR/raglight-request.json" > "$TMP_DIR/raglight.json" \
  || fail "RAGLight no respondió"
python3 - "$TMP_DIR/raglight.json" <<'PY' || fail "RAGLight respondió sin evidencia"
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p.get("executed_strategy") == "raglight", p
assert len(p.get("contexts", [])) > 0, p
PY
ok "RAGLight recupera evidencia"

# Evaluación completa opcional: puede ser lenta y LightRAG requiere proveedor externo.
if [[ "${RUN_EVALUATION,,}" == "true" ]]; then
  curl -fsS --max-time 900 \
    -X POST "$API_URL/api/v1/evaluations/run" \
    -H "Content-Type: application/json" \
    --data-binary @"$ROOT_DIR/infrastructure/requests/evaluation.json" > "$TMP_DIR/evaluation.json" \
    || fail "La evaluación completa falló"
  python3 - "$TMP_DIR/evaluation.json" <<'PY' || fail "La evaluación no devolvió métricas para los motores base"
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
s = p.get("summary", {})
for k in ("native_rag", "hybrid_rag", "raglight", "graphrag"):
    assert k in s, (k, s)
    assert float(s[k].get("successful_cases", 0)) > 0, (k, s[k])
PY
  ok "Evaluación completa devuelve métricas de los motores base"
fi

echo
echo "Resultado: $PASS comprobaciones OK, $WARN advertencias."
echo "La PoC supera la prueba funcional base."
