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
SEED_CONTAINER="${SEED_CONTAINER:-axiz-rag-dataset-seed}"
SEED_WAIT_SECONDS="${SEED_WAIT_SECONDS:-1200}"
OCR_TIMEOUT_SECONDS="${OCR_TIMEOUT_SECONDS:-600}"
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
ok "Frontend Angular está disponible"

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

# 3) OCR real: fuerza GLM-OCR sin indexar.
# La prueba OCR se aísla deliberadamente de Qdrant/Memgraph/RAGLight/LightRAG para que
# una GPU lenta no mezcle la latencia de inferencia con la validación de idempotencia.
OCR_STARTED="$(date +%s)"
set +e
OCR_HTTP_CODE="$(curl -sS --max-time "$OCR_TIMEOUT_SECONDS" \
  -o "$TMP_DIR/ocr.json" -w "%{http_code}" \
  -X POST "$API_URL/api/v1/documents/ingest?ocr=glm&index=false" \
  -F "file=@$OCR_SAMPLE")"
OCR_CURL_RC=$?
set -e
OCR_ELAPSED=$(( $(date +%s) - OCR_STARTED ))
if [[ "$OCR_CURL_RC" -eq 28 ]]; then
  fail "GLM-OCR excedió ${OCR_TIMEOUT_SECONDS}s en la prueba OCR (transcurridos: ${OCR_ELAPSED}s)"
elif [[ "$OCR_CURL_RC" -ne 0 ]]; then
  fail "La prueba OCR no pudo completar la llamada al backend (curl rc=$OCR_CURL_RC)"
fi
if [[ "$OCR_HTTP_CODE" -lt 200 || "$OCR_HTTP_CODE" -ge 300 ]]; then
  echo "Respuesta OCR (HTTP $OCR_HTTP_CODE):" >&2
  cat "$TMP_DIR/ocr.json" >&2 || true
  echo >&2
  fail "La prueba OCR real falló"
fi
python3 - "$TMP_DIR/ocr.json" <<'PY' || fail "La respuesta OCR no confirma uso de GLM-OCR"
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p.get("processor") == "glm-ocr", p
assert p.get("ocr_used") is True, p
assert p.get("indexed") is False, p
assert p.get("idempotency_key") == p.get("document_id"), p
assert int(p.get("chunks", 0)) > 0, p
assert float((p.get("timings_ms") or {}).get("parse_ocr", 0)) > 0, p
PY
ok "GLM-OCR procesó una imagen real en ${OCR_ELAPSED}s (prueba aislada, sin indexación)"

# 4) Antes de probar recuperación, el dataset base debe haber terminado de indexarse.
# La UI y el backend pueden usarse durante el bootstrap; las pruebas RAG requieren el corpus listo.
if command -v docker >/dev/null 2>&1 && docker inspect "$SEED_CONTAINER" >/dev/null 2>&1; then
  echo "Esperando dataset-seed (máximo ${SEED_WAIT_SECONDS}s)..."
  SEED_STARTED="$(date +%s)"
  LAST_NOTICE=0
  while true; do
    SEED_STATUS="$(docker inspect -f '{{.State.Status}}' "$SEED_CONTAINER" 2>/dev/null || echo unknown)"
    SEED_EXIT="$(docker inspect -f '{{.State.ExitCode}}' "$SEED_CONTAINER" 2>/dev/null || echo -1)"
    if [[ "$SEED_STATUS" == "exited" ]]; then
      [[ "$SEED_EXIT" == "0" ]] || fail "dataset-seed terminó con exit code $SEED_EXIT"
      ok "dataset-seed terminó correctamente"
      break
    fi
    SEED_ELAPSED=$(( $(date +%s) - SEED_STARTED ))
    if (( SEED_ELAPSED >= SEED_WAIT_SECONDS )); then
      fail "dataset-seed sigue en estado $SEED_STATUS después de ${SEED_WAIT_SECONDS}s"
    fi
    if (( SEED_ELAPSED - LAST_NOTICE >= 30 )); then
      echo "  dataset-seed sigue $SEED_STATUS (${SEED_ELAPSED}s)..."
      LAST_NOTICE=$SEED_ELAPSED
    fi
    sleep 5
  done
else
  warn "No se pudo inspeccionar $SEED_CONTAINER; se continúa asumiendo que el corpus ya fue precargado"
fi

# 5) Ingesta asíncrona: el request debe devolver 202/job y el worker completar el pipeline.
IDEMPOTENCY_SAMPLE="$ROOT_DIR/datasets/sample_documents/idempotency.md"
[[ -f "$IDEMPOTENCY_SAMPLE" ]] || fail "No existe la muestra de idempotencia: $IDEMPOTENCY_SAMPLE"
ASYNC_CODE="$(curl -sS --max-time 30 -o "$TMP_DIR/job-submit.json" -w "%{http_code}" \
  -X POST "$API_URL/api/v1/documents/jobs?index_external=false" \
  -F "file=@$IDEMPOTENCY_SAMPLE")"
[[ "$ASYNC_CODE" == "202" ]] || fail "La ingesta asíncrona no devolvió HTTP 202 (HTTP $ASYNC_CODE)"
JOB_ID="$(python3 - "$TMP_DIR/job-submit.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
assert p.get("status") == "queued", p
print(p["job_id"])
PY
)"
for _ in $(seq 1 60); do
  curl -fsS --max-time 15 "$API_URL/api/v1/jobs/$JOB_ID" > "$TMP_DIR/job.json" \
    || fail "No se pudo consultar el job asíncrono"
  JOB_STATUS="$(python3 - "$TMP_DIR/job.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))
PY
)"
  [[ "$JOB_STATUS" == "succeeded" ]] && break
  [[ "$JOB_STATUS" == "failed" ]] && { cat "$TMP_DIR/job.json" >&2; fail "El job asíncrono falló"; }
  sleep 1
done
python3 - "$TMP_DIR/job.json" <<'PY' || fail "El job no confirmó progreso/resultado"
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
assert p.get("status") == "succeeded", p
assert int(p.get("progress", 0)) == 100, p
assert (p.get("result") or {}).get("indexed") is True, p
PY
ok "Ingesta asíncrona completa un job sin bloquear el request HTTP"

# 6) Idempotencia documental: reingestar el mismo archivo debe reemplazar, no duplicar.
for attempt in 1 2; do
  curl -fsS --max-time 60 \
    -X POST "$API_URL/api/v1/documents/ingest?index_external=false" \
    -F "file=@$IDEMPOTENCY_SAMPLE" > "$TMP_DIR/idempotency-${attempt}.json" \
    || fail "La reingesta idempotente falló en el intento $attempt"
done
python3 - "$TMP_DIR/idempotency-1.json" "$TMP_DIR/idempotency-2.json" <<'PY' \
  || fail "La API no conservó la misma clave de idempotencia documental"
import json, sys
first = json.load(open(sys.argv[1], encoding="utf-8"))
second = json.load(open(sys.argv[2], encoding="utf-8"))
assert first.get("document_id") == second.get("document_id"), (first, second)
assert first.get("idempotency_key") == second.get("idempotency_key") == first.get("document_id")
assert second.get("replaced_existing") is True, second
assert int(second.get("chunks", 0)) > 0, second
PY
DOC_ID="$(python3 - "$TMP_DIR/idempotency-2.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["document_id"])
PY
)"
curl -fsS --max-time 30 "$API_URL/api/v1/documents/$DOC_ID/index-state" \
  > "$TMP_DIR/idempotency-state.json" \
  || fail "No se pudo verificar el estado de índices del documento idempotente"
python3 - "$TMP_DIR/idempotency-2.json" "$TMP_DIR/idempotency-state.json" <<'PY' \
  || fail "Qdrant/Memgraph/canónico contienen duplicados tras la reingesta"
import json, sys
response = json.load(open(sys.argv[1], encoding="utf-8"))
state = json.load(open(sys.argv[2], encoding="utf-8"))
chunks = int(response.get("chunks", 0))
assert state.get("consistent") is True, state
assert int(state.get("expected_chunks", -1)) == chunks, (response, state)
assert int(state.get("qdrant_chunks", -1)) == chunks, (response, state)
assert int(state.get("memgraph_chunks", -1)) == chunks, (response, state)
assert int(state.get("canonical_files", -1)) == 1, state
PY
ok "Reingesta idempotente reemplaza el documento sin duplicar Qdrant/Memgraph/canónico"

# 6) Native RAG: debe recuperar el documento de idempotencia ya precargado.
cat > "$TMP_DIR/native-request.json" <<'JSON'
{
  "question": "¿Cómo evita la idempotencia cobros duplicados cuando hay timeout?",
  "strategy": "native_rag",
  "top_k": 5,
  "include_trace": true,
  "include_evidence": true,
  "rerank": true,
  "deduplicate": true
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
ranking = (p.get("trace") or {}).get("ranking") or {}
assert ranking.get("reranked") is True, ranking
assert ranking.get("deduplicated") is True, ranking
assert int(ranking.get("candidates", 0)) >= len(p.get("contexts", [])), ranking
PY
ok "Native RAG recupera idempotency.md con reranking + deduplicación"

# Chat SSE por el mismo proxy Nginx usado por Angular. Debe exponer actividad y deltas
# antes de cerrar con el payload completo.
curl -sS -N --max-time 120 \
  -X POST "$FRONTEND_URL/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  --data-binary @"$TMP_DIR/native-request.json" > "$TMP_DIR/chat-sse.txt" \
  || fail "El chat SSE no respondió a través de Nginx"
grep -q '^event: stage' "$TMP_DIR/chat-sse.txt" \
  || fail "El stream SSE no publicó eventos de actividad"
grep -q '^event: retrieval' "$TMP_DIR/chat-sse.txt" \
  || fail "El stream SSE no publicó el resultado de retrieval"
grep -q '^event: delta' "$TMP_DIR/chat-sse.txt" \
  || fail "El stream SSE no publicó deltas de texto"
grep -q '^event: complete' "$TMP_DIR/chat-sse.txt" \
  || fail "El stream SSE no publicó el evento complete"
ok "Chat SSE publica actividad, retrieval, texto incremental y cierre por Nginx"

# 7) Adaptive Routing: la consulta relacional debe ir a GraphRAG.
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

# 8) Grafo: debe haber al menos una relación para AUTORIZACION.
curl -fsS --max-time 30 "$API_URL/api/v1/graph/neighborhood/AUTORIZACION" > "$TMP_DIR/graph.json" \
  || fail "La consulta de vecindad de Memgraph falló"
python3 - "$TMP_DIR/graph.json" <<'PY' || fail "Memgraph respondió sin relaciones para AUTORIZACION"
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(p.get("edges", [])) > 0, p
PY
ok "Memgraph devuelve relaciones para AUTORIZACION"

curl -fsS --max-time 30 "$API_URL/api/v1/graph/overview?limit=40" > "$TMP_DIR/graph-overview.json" \
  || fail "La vista general del grafo falló"
python3 - "$TMP_DIR/graph-overview.json" <<'PY' || fail "El overview de Memgraph no devolvió nodos/relaciones"
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
assert len(p.get("nodes", [])) > 0, p
assert len(p.get("edges", [])) > 0, p
PY
ok "API de administración del grafo devuelve overview navegable"

# 10) RAGLight real: debe recuperar al menos un contexto.
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

# 10) LightRAG real: si está configurado, debe recuperar contexto del corpus precargado.
if python3 - "$TMP_DIR/health.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if p.get("lightrag") else 1)
PY
then
  cat > "$TMP_DIR/lightrag-request.json" <<'JSON'
{
  "question": "¿Cómo evita la idempotencia cobros duplicados cuando hay timeout?",
  "strategy": "lightrag",
  "top_k": 5,
  "include_trace": true,
  "include_evidence": true
}
JSON
  curl -fsS --max-time 330 \
    -X POST "$API_URL/api/v1/query" \
    -H "Content-Type: application/json" \
    --data-binary @"$TMP_DIR/lightrag-request.json" > "$TMP_DIR/lightrag.json" \
    || fail "LightRAG está configurado pero no respondió dentro del timeout"
  python3 - "$TMP_DIR/lightrag.json" <<'PY' || fail "LightRAG respondió sin contexto utilizable"
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p.get("executed_strategy") == "lightrag", p
assert len(p.get("contexts", [])) > 0, p
PY
  ok "LightRAG recupera contexto real"
else
  warn "LightRAG no está configurado; se omite su consulta funcional"
fi

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
for k in ("native_rag", "hybrid_rag", "raglight", "graphrag", "auto"):
    assert k in s, (k, s)
    assert float(s[k].get("successful_cases", 0)) > 0, (k, s[k])
    for metric in ("mrr", "raw_mrr", "ndcg_at_k", "raw_ndcg_at_k", "recall_at_k"):
        assert metric in s[k], (k, metric, s[k])
comparison = p.get("ranking_comparison", [])
assert len(comparison) >= 5, comparison
assert all("delta_mrr" in row and "delta_ndcg_at_k" in row for row in comparison), comparison
assert int(p.get("dataset_cases", 0)) >= 10, p.get("dataset_cases")
PY
  ok "Evals comparan ranking raw vs reranked con MRR/nDCG/recall"
fi

echo
echo "Resultado: $PASS comprobaciones OK, $WARN advertencias."
echo "La PoC supera la prueba funcional base."
