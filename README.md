# Adaptive RAG & GraphRAG para Payment Knowledge

Esta PoC prueba una arquitectura de conocimiento documental para payment processing. El objetivo no es hacer otro chat con PDFs, sino comparar varias formas de recuperar información sobre el mismo corpus y poder ver por qué una estrategia funciona mejor que otra según el tipo de pregunta.

La aplicación procesa documentos con Docling, usa GLM-OCR como ruta OCR cuando el contenido viene escaneado o la extracción normal no alcanza un mínimo de texto, construye una representación vectorial en Qdrant y un grafo de conocimiento en Memgraph, y expone cinco estrategias de recuperación: Native RAG, Hybrid RAG, RAGLight, GraphRAG y LightRAG. En modo `auto`, un router decide entre recuperación semántica, híbrida o GraphRAG según señales observables de la consulta. RAGLight corre en un servicio Python separado para aislar su árbol de dependencias del backend principal y poder mantener Docling y RAGLight en versiones actuales sin forzar paquetes incompatibles en el mismo entorno.


> Versión v13: corrige la atribución de fuentes en RAGLight y LightRAG para que Hit Rate/MRR midan documentos reales, elimina el `temperature` forzado en la síntesis OpenAI-compatible y hace que la exploración de vecindad de Memgraph tolere mayúsculas y acentos.

La interfaz está hecha en Angular tomando como base visual el proyecto de ejemplo: navegación a la izquierda, conversación al centro y configuración/evaluación a la derecha. La adapté para carga documental, selección de estrategia, evidencia recuperada, trazabilidad técnica, evaluación y exploración del grafo.

## Qué quiero demostrar con esta PoC

El caso funcional está centrado en soporte técnico y operativo de pagos. El corpus de ejemplo incluye autorización ISO 8583, códigos de rechazo, conciliación, idempotencia, tokenización y controles PCI. Algunas preguntas se resuelven mejor por similitud semántica, otras necesitan conservar términos exactos como `05` o `ISO 8583`, y otras requieren relacionar conceptos como autorización, adquirente y conciliación.

La PoC permite probar estos caminos sobre el mismo conjunto de documentos:

| Estrategia | Qué implementa | Cuándo aporta más |
| --- | --- | --- |
| Native RAG | embedding denso + búsqueda vectorial en Qdrant | preguntas semánticas |
| Hybrid RAG | ranking vectorial + ranking lexical + Reciprocal Rank Fusion | códigos, términos exactos y consultas mixtas |
| RAGLight | flujo alterno con RAGLight, Qdrant y búsqueda híbrida BM25 + semántica + RRF | comparar implementación propia contra framework |
| GraphRAG | búsqueda híbrida + entidades/relaciones y expansión de un salto en Memgraph | relaciones, dependencias y flujo de componentes |
| LightRAG | flujo graph-RAG alternativo usando LightRAG SDK | comparar otra estrategia basada en grafo |
| Adaptive Routing | reglas explícitas sobre señales de la pregunta | seleccionar estrategia sin que el cliente conozca la implementación |

La comparación no pretende declarar un ganador universal. El endpoint de evaluación mide `Hit Rate`, `MRR` y latencia de retrieval contra un pequeño set controlado incluido en `datasets/evaluation/questions.json`.

## Tecnologías y versiones

Las versiones principales están fijadas para evitar cambios sorpresivos. Docling y RAGLight se ejecutan en procesos Python distintos porque sus dependencias de CLI no son compatibles entre sí en las versiones seleccionadas: Docling 2.130.0 termina requiriendo `typer>=0.19,<0.27`, mientras RAGLight 3.4.7 fija `typer==0.16.0`. En vez de degradar una de las dos tecnologías, la PoC las aísla y las conecta por HTTP.

| Componente | Versión usada |
| --- | ---: |
| Python | 3.12+ y < 3.14 |
| FastAPI | 0.141.1 |
| Uvicorn | 0.54.0 |
| Pydantic | 2.13.5 |
| Pydantic Settings | 2.15.0 |
| Docling | 2.130.0 |
| GLM-OCR SDK | 0.1.5 |
| Qdrant Server | 1.18.3 |
| qdrant-client backend | 1.19.1 |
| qdrant-client RAGLight | 1.17.0 (dependencia fijada por RAGLight 3.4.7) |

La versión de Qdrant no está elegida al azar. RAGLight 3.4.7 exige `qdrant-client==1.17.0` y el backend usa `qdrant-client==1.19.1`; Qdrant Server `1.18.3` es el punto común compatible porque la diferencia de versión menor es de uno en ambos casos. No se fuerza una versión de cliente distinta dentro de RAGLight.
| Memgraph | 3.13.1 |
| Neo4j Python Driver | 6.3.1, usado solo como cliente Bolt para Memgraph |
| RAGLight | 3.4.7 |
| LangGraph en RAGLight | 1.0.5 |
| langgraph-prebuilt en RAGLight | 1.0.5 |
| langgraph-checkpoint en RAGLight | 3.0.1 |
| langgraph-sdk en RAGLight | 0.3.0 |
| LightRAG | 1.5.7 |
| Angular | 21.2.24 |
| TypeScript | 5.9.3 |
| Node para frontend | 22.x |
| Docker backend | `python:3.12-slim-bookworm` |
| Docker RAGLight | `python:3.12-slim-bookworm` |
| Docker frontend build | `node:22-bookworm-slim` |

El backend, procesamiento, retrieval, evaluación y scripts están escritos en Python. La única excepción es `frontend/`, porque Angular se implementa con TypeScript/HTML/CSS.

## Arquitectura

```mermaid
flowchart LR
    UI[Angular Frontend] --> API[FastAPI pe.axiz.payment_knowledge]

    subgraph Ingestion[Ingesta documental]
      D[PDF DOCX MD TXT Imagen] --> DOC[Docling]
      DOC --> CHECK{Texto suficiente}
      CHECK -- Sí --> CAN[Markdown canónico]
      CHECK -- No / escaneado --> OCR[GLM-OCR]
      OCR --> CAN
      CAN --> CH[Chunking estructural + entidades]
    end

    API --> Ingestion
    CH --> Q[(Qdrant)]
    CH --> M[(Memgraph)]
    CAN --> RLHTTP[RAGLight Adapter HTTP]
    RLHTTP --> RLS[RAGLight Service]
    RLS --> Q
    CAN --> LR[LightRAG index]

    API --> ROUTER[Adaptive Router]
    ROUTER --> NR[Native RAG]
    ROUTER --> HR[Hybrid RAG]
    ROUTER --> GR[GraphRAG]
    API --> RLHTTP
    API --> LRF[LightRAG]

    NR --> Q
    HR --> Q
    GR --> Q
    GR --> M
    LRF --> LR

    NR --> GEN[Answer Generator]
    HR --> GEN
    GR --> GEN
    RLHTTP --> GEN
    LRF --> GEN
    GEN --> API
    API --> UI

    API --> EV[Evaluation]
    EV --> NR
    EV --> HR
    EV --> RLHTTP
    EV --> GR
    EV --> LRF
```

Qdrant y Memgraph son los únicos componentes persistentes de infraestructura. Docker Compose también levanta el backend, el servicio aislado de RAGLight, el frontend y un contenedor de bootstrap que carga el dataset de ejemplo y termina cuando la ingesta concluye. No agregué PostgreSQL, Kafka, MongoDB, KurrentDB, InfluxDB ni Drools porque no aportan a la prueba técnica.

El servicio RAGLight no agrega una nueva base ni duplica infraestructura. Sigue usando Qdrant, pero tiene su propio runtime Python. Esa separación existe por compatibilidad de dependencias y también deja claro el límite entre el framework RAGLight y el backend de la PoC.

## Cómo funciona la ingesta

1. El archivo entra por `POST /api/v1/documents/ingest` o por la carga del dataset.
2. Para `.md` y `.txt` se conserva el contenido directamente.
3. Para PDF, DOCX y otros formatos soportados se usa Docling y se exporta a Markdown.
4. Si el archivo es una imagen o el texto extraído queda debajo de `OCR_MIN_TEXT_CHARS`, se deriva a GLM-OCR.
5. El documento normalizado se guarda temporalmente como Markdown canónico dentro de `.runtime/canonical`.
6. `SemanticChunker` agrupa bloques conservando estructura y un solape controlado.
7. `PaymentEntityExtractor` detecta conceptos de pagos y códigos que sirven para el grafo.
8. Los chunks se indexan en Qdrant.
9. Documentos, chunks, entidades y relaciones de co-ocurrencia se materializan en Memgraph.
10. Para una carga individual, el backend crea un directorio de staging con el Markdown canónico y RAGLight indexa ese directorio. Para el dataset inicial, primero se normalizan todos los documentos y después RAGLight indexa el directorio canónico completo en una sola operación.
11. Antes de la precarga del dataset se reinician únicamente las colecciones propias de RAGLight para evitar duplicados en un framework que genera IDs nuevos en cada ingesta.
12. Si LightRAG está configurado con un proveedor OpenAI-compatible, el contenido también se inserta en su índice.

El identificador de cada chunk es un UUID determinista para que Qdrant acepte el punto y para que reingestar el mismo contenido no cree identificadores distintos. Las relaciones `CO_OCCURS` de Memgraph también incluyen el `chunk_id`, de modo que una segunda carga no infla artificialmente los pesos.

## Estrategias de retrieval

### Native RAG

`NativeRagRetriever` genera el embedding de la pregunta y consulta la colección principal de Qdrant por similitud coseno. Es el camino más directo y sirve como baseline semántico.

### Hybrid RAG

`HybridRagRetriever` combina dos rankings independientes:

- dense retrieval en Qdrant;
- ranking lexical calculado sobre los chunks indexados;
- Reciprocal Rank Fusion para mezclar ambas listas sin depender de que los scores estén en la misma escala.

La intención es que consultas como `código 05` no pierdan el identificador exacto por depender solo del embedding.

### RAGLight

`RagLightAdapter` ya no importa el paquete RAGLight dentro del backend. Consume por HTTP un servicio Python independiente definido en `raglight_service/`. Ese servicio carga RAGLight 3.4.7, usa una colección Qdrant propia, embeddings Hugging Face y `SEARCH_HYBRID`. El framework combina BM25 + búsqueda semántica + RRF. La integración usa directorios porque `VectorStore.ingest(data_path=...)` de RAGLight espera una carpeta, no un archivo individual.

La separación es intencional. Docling 2.130.0 y RAGLight 3.4.7 no pueden convivir hoy en el mismo environment por el pin incompatible de `typer`. Mantenerlos en contenedores y entornos Conda distintos permite conservar las dos versiones sin `--no-deps`, sin forzar un `typer` incorrecto y sin degradar Docling.

Además fijo el conjunto LangGraph usado dentro de RAGLight (`langgraph==1.0.5`, `langgraph-prebuilt==1.0.5`, `langgraph-checkpoint==3.0.1`, `langgraph-sdk==0.3.0`). La razón es reproducibilidad: `langgraph-prebuilt` 1.0.x admite versiones más nuevas dentro de su rango, pero releases posteriores empezaron a importar `ExecutionInfo`/`ServerInfo` desde `langgraph.runtime`, APIs que no están disponibles en `langgraph==1.0.5`. Sin el pin, pip puede resolver una combinación que instala correctamente pero falla al importar RAGLight en runtime. El Dockerfile ejecuta un import de compatibilidad durante el build para detectar esta clase de error antes de levantar los contenedores.

RAGLight se usa como motor de retrieval alternativo y la generación sigue en la capa común. Así puedo comparar `Native/Hybrid implementado por nosotros` contra `Hybrid provisto por RAGLight` sin mezclar diferencias del modelo generativo. El servicio valida la ingesta comparando la cantidad de puntos de Qdrant antes y después; si RAGLight no agrega al menos un punto por documento canónico, devuelve error y el bootstrap falla en vez de marcar un falso positivo.

### GraphRAG

`GraphRagRetriever` fusiona dos fuentes:

- retrieval híbrido desde Qdrant;
- evidencia estructurada desde Memgraph.

En Memgraph se crean nodos `Document`, `Chunk` y `Entity`. Los chunks apuntan a las entidades mediante `MENTIONS`; cuando dos entidades aparecen en el mismo chunk se materializa `CO_OCCURS`. La búsqueda toma entidades semilla detectadas en la pregunta y expande un salto hacia entidades relacionadas y chunks conectados. Luego fusiona ese resultado con Hybrid RAG.

No estoy usando Neo4j como servidor. La dependencia `neo4j` del `pyproject.toml` es el driver Bolt de Python que se conecta a Memgraph.

### LightRAG

`LightRagAdapter` integra `lightrag-hku` como flujo alterno. Para construir entidades, relaciones y embeddings necesita un LLM y un endpoint de embeddings. La configuración usa una API OpenAI-compatible para no amarrar la PoC a un único proveedor.

Con `LIGHTRAG_ENABLED=true` y las variables `OPENAI_*` configuradas, la ingesta inserta el documento en LightRAG y las consultas usan `QueryParam(mode="hybrid", only_need_context=True)`. La capa común se encarga después de generar la respuesta final.

### Adaptive Routing

El router es intencionalmente explicable:

- términos de relaciones o flujo como `relación`, `componentes`, `entre`, `depende` -> `graphrag`;
- códigos o identificadores exactos como `05`, `51`, `91`, `ISO 8583`, `PAY-*`, `RC-*` -> `hybrid_rag`;
- el resto -> `native_rag`.

RAGLight y LightRAG quedan disponibles como estrategias manuales y dentro de la evaluación. De esta forma el modo `auto` no depende de que haya credenciales externas para funcionar.

## Generación de respuesta

Por defecto `LLM_PROVIDER=extractive`. La aplicación devuelve extractos de la evidencia recuperada y no necesita un LLM externo para probar retrieval, routing, Qdrant y Memgraph.

Si se quiere síntesis generativa:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5-mini
```

`OPENAI_BASE_URL` también puede apuntar a un servidor local compatible con OpenAI. Para LightRAG se usan además:

```env
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSION=1536
LIGHTRAG_ENABLED=true
```

Si el endpoint local requiere un token aunque no lo valide, se puede colocar un valor no vacío en `OPENAI_API_KEY`.

## GLM-OCR

La PoC usa GLM-OCR **self-hosted por defecto**. No necesito saldo ni `ZHIPU_API_KEY`: Docker Compose levanta un servicio `glm-ocr` basado en vLLM y el modelo `zai-org/GLM-OCR`. El backend mantiene el pipeline completo del SDK `glmocr[selfhosted]` y usa la GPU solo para la inferencia del modelo; el detector de layout se deja en CPU para reservar VRAM.

Configuración usada dentro de Docker:

```env
GLM_OCR_MODE=selfhosted
GLM_OCR_API_URL=http://glm-ocr:8080/v1/chat/completions
GLM_OCR_MODEL=glm-ocr
GLM_OCR_LAYOUT_DEVICE=cpu
GLM_OCR_MAX_WORKERS=4
GLM_OCR_REQUEST_TIMEOUT_SECONDS=300
OCR_POLICY=auto
ZHIPU_API_KEY=
```

La política OCR puede elegirse por request:

```text
auto     imágenes -> GLM-OCR; PDF -> Docling y fallback a GLM-OCR si el texto es insuficiente
glm      fuerza GLM-OCR y omite Docling
docling  fuerza Docling y no llama a GLM-OCR
```

Ejemplo para forzar GLM-OCR sobre un PDF escaneado:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/ingest?ocr=glm" \
  -F "file=@datasets/ocr_samples/documento_escaneado_prueba_ocr_pagos.pdf"
```

El contenedor usa la imagen oficial de vLLM `v0.19.0-ubuntu2404`, con `transformers>=5.3.0`, tal como requiere la guía de despliegue de GLM-OCR. La imagen de vLLM expone el intérprete como `python3`, por lo que `Dockerfile.glm-ocr` instala/valida `transformers==5.3.0` con `python3 -m pip`; no depende del alias `python`. Se limita el contexto a 16384, la concurrencia OCR a 4 workers y la utilización de VRAM a `0.75` para una PoC de baja concurrencia.

## Estructura del proyecto

```text
.
├── backend/
│   ├── Dockerfile             imagen Python directa, sin Conda
│   ├── src/pe/axiz/payment_knowledge/
│   │   ├── api/              endpoints FastAPI
│   │   ├── application/      orquestación de ingesta, consulta y evaluación
│   │   ├── domain/           contratos y modelos Pydantic
│   │   ├── generation/       respuesta extractiva u OpenAI-compatible
│   │   ├── infrastructure/   Docling, GLM-OCR, Qdrant, Memgraph y embeddings
│   │   └── retrieval/        Native, Hybrid, GraphRAG, LightRAG, adapter RAGLight y router
│   └── tests/                pruebas unitarias del backend
├── raglight_service/
│   ├── Dockerfile             runtime Python aislado de RAGLight
│   ├── environment.yml       entorno Conda local exclusivo de RAGLight
│   ├── pyproject.toml        dependencias del servicio RAGLight
│   ├── src/pe/axiz/raglight_service/
│   │   └── main.py           API interna de indexación y búsqueda
│   └── tests/                pruebas del servicio aislado
├── datasets/
│   ├── sample_documents/     corpus de payment processing
│   ├── ocr_samples/          PDF/JPG escaneados para validar GLM-OCR local
│   ├── evaluation/           preguntas y fuentes esperadas
│   └── seed.py               carga del dataset usando la API
├── scripts/
│   └── verify.py             validaciones reproducibles
├── infrastructure/
│   ├── Dockerfile.ai         base reusable para backend y RAGLight
│   ├── Dockerfile.glm-ocr    extensión de vLLM para GLM-OCR local
│   └── docker-compose.yml    stack completo
├── frontend/                 Angular, basado visualmente en la interfaz de referencia
│   ├── Dockerfile             build con imagen oficial de Node, sin Conda
│   └── proxy.conf.json       proxy local /api hacia FastAPI
├── infrastructure/
│   ├── docker-compose.yml    stack completo: stores, RAGLight, API, seed y frontend
│   ├── requests/             payloads de ejemplo
│   └── responses/            respuestas de referencia de la API
├── .env.example
├── environment.yml          entorno Conda local del backend + frontend
├── pyproject.toml           dependencias del backend principal
└── README.md
```

Toda la explicación del proyecto está en este README. No hay documentación duplicada en otras carpetas.

## Código principal

Los puntos que conviene revisar primero son:

| Archivo | Responsabilidad |
| --- | --- |
| `backend/src/pe/axiz/payment_knowledge/main.py` | crea FastAPI y CORS |
| `container.py` | arma las dependencias y los cinco motores |
| `application/services.py` | coordina ingesta, query y evaluación |
| `infrastructure/document_processing.py` | Docling, GLM-OCR, chunking y entidades |
| `infrastructure/qdrant_store.py` | persistencia y consulta vectorial |
| `infrastructure/memgraph_store.py` | nodos, relaciones, expansión y exploración del grafo |
| `retrieval/native.py` | Native RAG e Hybrid RAG con RRF |
| `retrieval/raglight_adapter.py` | cliente HTTP hacia el servicio RAGLight aislado |
| `retrieval/graphrag.py` | fusión de evidencia híbrida y grafo |
| `retrieval/lightrag_adapter.py` | integración con LightRAG |
| `retrieval/router.py` | decisión del modo `auto` |
| `generation/llm.py` | generación común para que los motores se comparen con la misma salida |
| `raglight_service/src/pe/axiz/raglight_service/main.py` | RAGLight, búsqueda híbrida y colección Qdrant dedicada |
| `frontend/src/app/app.ts` | interacción de la UI con la API |
| `frontend/src/app/app.html` | layout de conversación, configuración, métricas y grafo |

## Endpoints en orden de uso

| Orden | Método y endpoint | Uso funcional | Qué hace técnicamente |
| ---: | --- | --- | --- |
| 1 | `GET /api/v1/health` | verificar que la PoC está lista | prueba Qdrant, Memgraph, RAGLight, LightRAG y el servicio local GLM-OCR |
| 2 | `POST /api/v1/datasets/seed` | cargar el dataset de ejemplo | recorre `datasets/sample_documents`, procesa, fragmenta e indexa en los motores habilitados |
| 2b | `POST /api/v1/documents/ingest?ocr=auto|glm|docling` | cargar un documento propio | multipart upload -> política OCR -> chunks -> Qdrant/Memgraph/RAGLight/LightRAG |
| 3 | `POST /api/v1/query` | consultar conocimiento | ejecuta estrategia solicitada o router `auto`, recupera evidencia y genera respuesta |
| 4 | `GET /api/v1/graph/neighborhood/{entity}` | revisar relaciones | consulta relaciones `CO_OCCURS` en Memgraph |
| 5 | `POST /api/v1/evaluations/run` | comparar motores | ejecuta retrieval contra el set de evaluación y calcula Hit Rate, MRR y latencia |
| - | `GET /docs` | probar API desde navegador | Swagger generado por FastAPI |

Los JSON usados en los ejemplos están en `infrastructure/requests` y `infrastructure/responses`.

El servicio RAGLight expone cinco endpoints internos. El frontend no los consume directamente; el backend actúa como fachada.

| Método y endpoint | Uso técnico |
| --- | --- |
| `GET http://raglight-service:8010/health` | comprobar que el proceso HTTP está disponible |
| `POST http://raglight-service:8010/reset` | limpiar las colecciones propias de RAGLight antes de una precarga determinista |
| `GET http://raglight-service:8010/ready` | inicializar embeddings y Qdrant, validando credenciales/descarga antes de la ingesta |
| `POST http://raglight-service:8010/index` | indexar un directorio de documentos canónicos compartido por volumen y verificar que se creen puntos |
| `POST http://raglight-service:8010/search` | ejecutar retrieval híbrido RAGLight y devolver evidencia al backend |

## Variables del `.env` en Docker

El archivo `.env` está en la raíz del proyecto, mientras que el Compose vive en `infrastructure/`. Como se ejecuta con `-f infrastructure/docker-compose.yml`, Docker Compose considera `infrastructure/` como directorio del proyecto para la interpolación automática. Para evitar que un `HF_TOKEN`, `OPENAI_API_KEY` u otra variable del `.env` raíz quede fuera de los contenedores, esta versión declara `env_file: ../.env` explícitamente en `backend`, `raglight-service` y `glm-ocr`.

Las URLs que cambian dentro de Docker (`QDRANT_URL`, `MEMGRAPH_URI`, `RAGLIGHT_SERVICE_URL`, `LIGHTRAG_WORKDIR` y `DATASETS_DIR`) se sobrescriben después con valores internos de la red Compose. Las credenciales y switches del `.env` no se pisan con valores vacíos.

Además, backend, RAGLight y GLM-OCR comparten un volumen de cache de Hugging Face y usan `HF_HOME`, `HUGGINGFACE_HUB_CACHE` y `SENTENCE_TRANSFORMERS_HOME`. El token nunca se imprime; los endpoints de salud solo muestran `hf_token_configured=true/false`.

Para confirmar que el token llegó al contenedor sin mostrarlo:

```bash
docker compose -f infrastructure/docker-compose.yml exec backend python -c 'import os; print(bool(os.getenv("HF_TOKEN")))'
docker compose -f infrastructure/docker-compose.yml exec raglight-service python -c 'import os; print(bool(os.getenv("HF_TOKEN")))'
```

También se puede consultar:

```bash
curl -s http://localhost:8000/api/v1/health | python -m json.tool
curl -s http://localhost:8010/health | python -m json.tool
```

RAGLight incorpora además `GET /ready`: inicializa el modelo de embeddings y Qdrant antes de procesar el dataset. Si falla una descarga, autenticación o inicialización, el bootstrap termina temprano con el error real en vez de esperar hasta `/index`. Los errores HTTP del servicio conservan ahora el `detail` original y el servicio registra el traceback completo. El endpoint `GET /health` también informa las versiones efectivas de RAGLight, LangGraph, `langgraph-prebuilt`, checkpoint, SDK y qdrant-client para que una incompatibilidad transitiva sea visible sin entrar al contenedor.


## Requisitos

Hay dos formas de ejecutar la PoC. La recomendada es Docker porque deja listo el stack completo y también precarga el dataset. Conda queda como alternativa para desarrollo local.

Para Docker necesito:

- Docker Engine o Docker Desktop con Compose v2;
- una GPU NVIDIA visible desde Docker (`nvidia-smi` en WSL/Linux y soporte `--gpus all`);
- aproximadamente 8 GB de VRAM como referencia práctica para GLM-OCR 0.9B en esta PoC;
- 16 GB de RAM del sistema como mínimo razonable y 32 GB recomendados para ejecutar todo el stack con margen.

El primer arranque descarga los pesos de GLM-OCR desde Hugging Face y puede tardar varios minutos. Los pesos y la caché de compilación de vLLM quedan persistidos en volúmenes Docker.

Para desarrollo local necesito además una distribución Conda, por ejemplo Miniconda o Miniforge. No uso `venv` y no hay pasos manuales de `pip install`. Python y Node salen del entorno Conda.

GLM-OCR local no necesita ninguna credencial de Zhipu. Las credenciales externas solo hacen falta para LightRAG y para síntesis generativa cuando uso el adapter OpenAI-compatible.

## Ejecución recomendada: todo con Docker

Desde la raíz del proyecto:

```bash
docker compose -f infrastructure/docker-compose.yml up --build
```

Ese comando es suficiente para la ejecución Docker completa: levanta la infraestructura, backend y RAGLight, ejecuta `dataset-seed` automáticamente y arranca el frontend cuando el seed termina con código 0. No hace falta ejecutar manualmente `POST /api/v1/datasets/seed`; hacerlo de nuevo vuelve a procesar el corpus y, con LightRAG habilitado, repite llamadas al proveedor LLM/embeddings. Los `curl` siguientes son verificaciones funcionales, no pasos adicionales de arranque.

Ese único comando hace el bootstrap completo:

1. levanta Qdrant;
2. levanta Memgraph;
3. construye y levanta `raglight-service` con Python 3.12 y RAGLight 3.4.7;
4. construye y levanta el backend principal con Python 3.12, Docling, GLM-OCR y LightRAG;
5. ejecuta `dataset-seed` como contenedor de una sola corrida usando la misma definición de build del backend;
6. `dataset-seed` espera como máximo 900 segundos solo por la disponibilidad de backend/Qdrant/Memgraph y luego llama una única vez a `POST /api/v1/datasets/seed`;
7. se crean chunks, vectores, nodos y relaciones; RAGLight reinicia sus colecciones e indexa el directorio canónico completo una sola vez;
8. `dataset-seed` termina con código `0` solo si las rutas obligatorias habilitadas concluyen correctamente;
9. cuando el seed termina correctamente, levanta el frontend Angular.

No hay que ejecutar inserts ni scripts de carga después. `dataset-seed` termina con código `0` cuando la precarga finaliza; eso es esperado. El límite de 900 segundos aplica únicamente a la espera inicial de infraestructura, no al procesamiento de LightRAG/RAGLight. La llamada de indexación no tiene timeout de lectura por defecto, porque la primera construcción de embeddings y grafo puede durar bastante más de 15 minutos. Si una ruta obligatoria habilitada falla, el backend devuelve error, `dataset-seed` termina distinto de cero y el frontend no arranca como si la PoC estuviera completa.

### Compatibilidad LangGraph dentro de RAGLight

Si aparece `ImportError: cannot import name 'ExecutionInfo' from 'langgraph.runtime'`, no es un problema de `HF_TOKEN`: significa que `langgraph-prebuilt` quedó más nuevo que el `langgraph` compatible con RAGLight. Esta versión fija el conjunto probado y además valida el import durante el build. Si esa validación falla, Docker debe detenerse en la construcción de `raglight-service` en vez de arrancar un contenedor roto.

Para comprobar las versiones en ejecución:

```bash
curl -s http://localhost:8010/health | python -m json.tool
```

Debe mostrar, entre otros valores, `raglight=3.4.7`, `langgraph=1.0.5` y `langgraph-prebuilt=1.0.5`.

### Por qué backend y RAGLight se construyen por separado

El error `ResolutionImpossible` que aparece al intentar instalar todo junto no se resuelve forzando pip. Docling 2.130.0 depende de un rango de Typer que empieza en 0.19, mientras RAGLight 3.4.7 fija Typer 0.16.0. Por eso el Compose usa dos imágenes Python distintas. El backend no instala `raglight`; `raglight-service` no instala Docling. Los dos comparten Qdrant y el volumen `.runtime`.

`dataset-seed` declara el mismo `image` y el mismo `build` que `backend` mediante un ancla YAML. Así Compose sabe construir la imagen local y no necesita buscar `axiz-adaptive-rag-backend:local` en Docker Hub. No hace falta `docker login`.

Dentro de Docker no uso Conda ni Micromamba. Los servicios Python parten de `python:3.12-slim-bookworm` y se instalan directamente con `python -m pip install .`. El frontend se construye con `node:22-bookworm-slim` y luego se sirve desde Nginx. Conda queda reservado para desarrollo local desde el IDE.

Servicios expuestos:

| Servicio | URL / puerto | Uso |
| --- | --- | --- |
| Frontend | `http://localhost:4200` | interfaz Angular |
| API | `http://localhost:8000` | FastAPI principal |
| RAGLight service | `http://localhost:8010` | API interna de retrieval RAGLight; expuesta para diagnóstico local |
| Swagger | `http://localhost:8000/docs` | prueba de endpoints |
| Qdrant HTTP | `localhost:6333` | vectores y colecciones |
| Qdrant gRPC | `localhost:6334` | interfaz gRPC |
| Memgraph Bolt | `localhost:7687` | acceso al grafo |
| Memgraph | `localhost:7444` | puerto expuesto por Memgraph |

Para revisar el estado en otra terminal:

```bash
docker compose -f infrastructure/docker-compose.yml ps
```

Para detener el stack se usa `Ctrl+C`. Si luego quiero remover los contenedores sin borrar los datos:

```bash
docker compose -f infrastructure/docker-compose.yml down
```

Para resetear completamente la PoC, incluidas las colecciones, el grafo y el runtime de LightRAG:

```bash
docker compose -f infrastructure/docker-compose.yml down -v
```

No configuré health checks periódicos en Compose. El bootstrap usa reintentos finitos solo hasta que el dataset logra cargarse, así que no hay llamadas de health ejecutándose permanentemente ni ensuciando los logs.

### Configuración del `.env` y credenciales

La PoC base puede levantar sin credenciales externas. En ese modo funcionan Docling para documentos digitales, Native RAG, Hybrid RAG, RAGLight, GraphRAG, Adaptive Routing, Qdrant, Memgraph, evaluación y el frontend. La generación queda en modo extractivo.

Creo el archivo desde la raíz solo cuando quiero activar las capacidades que dependen de servicios externos:

```bash
cp .env.example .env
```

Para Docker, Compose usa las variables del `.env` ubicado en la raíz. Las direcciones internas de Qdrant y Memgraph ya están definidas en `infrastructure/docker-compose.yml`, por eso no tengo que cambiar sus URLs para ejecutar el stack completo.

`CORS_ORIGINS` se mantiene como texto para evitar que `pydantic-settings` intente decodificarlo como JSON antes de aplicar la configuración. Acepto ambas formas: `http://localhost:4200,http://localhost:8080` o `["http://localhost:4200","http://localhost:8080"]`. Para esta PoC basta `http://localhost:4200`.

| Variable | ¿Obligatoria? | Para qué sirve | De dónde sale |
| --- | --- | --- | --- |
| `LLM_PROVIDER` | No | `extractive` no llama a ningún LLM; `openai` activa síntesis generativa | valor de configuración |
| `OPENAI_API_KEY` | Solo para generación OpenAI y LightRAG con este adapter | autenticación del LLM y embeddings usados por LightRAG | colocar la clave del proveedor configurado |
| `OPENAI_BASE_URL` | Solo si cambio de endpoint | endpoint OpenAI-compatible | URL del proveedor; el ejemplo deja el endpoint estándar |
| `OPENAI_MODEL` | Solo con `LLM_PROVIDER=openai` o LightRAG | modelo generativo | nombre soportado por el proveedor |
| `OPENAI_EMBEDDING_MODEL` | Para LightRAG | embeddings de LightRAG | nombre soportado por el proveedor |
| `OPENAI_EMBEDDING_DIMENSION` | Para LightRAG | dimensión que debe coincidir con el embedding elegido | documentación del modelo de embeddings |
| `GLM_OCR_MODE` | No en Docker | modo del SDK; Compose fuerza `selfhosted` | `selfhosted` |
| `GLM_OCR_API_URL` | No en Docker | endpoint OpenAI-compatible del contenedor vLLM | `http://glm-ocr:8080/v1/chat/completions` |
| `GLM_OCR_MODEL` | No en Docker | nombre servido por vLLM | `glm-ocr` |
| `GLM_OCR_LAYOUT_DEVICE` | No en Docker | dispositivo del detector de layout del SDK | `cpu` |
| `GLM_OCR_MAX_WORKERS` | No en Docker | concurrencia máxima del pipeline OCR | `4` |
| `GLM_OCR_REQUEST_TIMEOUT_SECONDS` | No en Docker | timeout por llamada al modelo self-hosted | `300` |
| `OCR_POLICY` | No | política por defecto de ingesta | `auto`, `glm` o `docling` |
| `ZHIPU_API_KEY` | No | queda solo por compatibilidad si manualmente vuelvo a MaaS | vacío en esta PoC |
| `RAGLIGHT_ENABLED` | No | habilita el flujo RAGLight | `true`/`false`; no necesita token |
| `RAGLIGHT_SERVICE_URL` | No | URL del servicio aislado RAGLight | local: `http://localhost:8010`; Docker la sobrescribe a `http://raglight-service:8010` |
| `RAGLIGHT_TIMEOUT_SECONDS` | No | timeout de indexación/búsqueda RAGLight | `600` por defecto |
| `LIGHTRAG_ENABLED` | No | habilita el flujo LightRAG | `true`/`false`; además necesita LLM + embeddings |
| `QDRANT_*` | No para Docker | colección y conexión vectorial | la PoC levanta Qdrant local |
| `MEMGRAPH_*` | No para Docker | conexión al grafo | la PoC levanta Memgraph local sin usuario/password |
| `EMBEDDING_*` | No | embeddings usados por Native/Hybrid/RAGLight | modelo público configurado; no necesita token |
| `HF_TOKEN` | No | autenticación opcional contra Hugging Face Hub | evita el warning de acceso anónimo y puede mejorar límites/velocidad de descarga |

Docling no requiere API key. RAGLight tampoco requiere API key en esta PoC porque usa embeddings públicos/Hugging Face y Qdrant local. `HF_TOKEN` es opcional: sin él la descarga funciona de forma anónima, pero Hugging Face puede mostrar un warning y aplicar límites más bajos. Qdrant y Memgraph tampoco necesitan tokens con la configuración del Compose.

GLM-OCR está incorporado al Compose y no usa `ZHIPU_API_KEY`. `HF_TOKEN` sigue siendo opcional, aunque ayuda a evitar límites anónimos al descargar los pesos desde Hugging Face.

### Ejemplo mínimo

```env
LLM_PROVIDER=extractive
RAGLIGHT_ENABLED=true
LIGHTRAG_ENABLED=true
GLM_OCR_MODE=selfhosted
GLM_OCR_API_URL=http://localhost:8080/v1/chat/completions
GLM_OCR_MODEL=glm-ocr
GLM_OCR_LAYOUT_DEVICE=cpu
GLM_OCR_MAX_WORKERS=4
GLM_OCR_REQUEST_TIMEOUT_SECONDS=300
OCR_POLICY=auto
ZHIPU_API_KEY=
OPENAI_API_KEY=
```

Con esta configuración el OCR funciona localmente. LightRAG queda inactivo si no configuro el proveedor OpenAI-compatible; Native RAG, Hybrid RAG, RAGLight, GraphRAG, Adaptive Routing, Docling y GLM-OCR siguen disponibles.

### Ejemplo con generación y LightRAG habilitados

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-ejemplo-reemplazar
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSION=1536

GLM_OCR_MODE=selfhosted
GLM_OCR_API_URL=http://localhost:8080/v1/chat/completions
GLM_OCR_MODEL=glm-ocr
GLM_OCR_LAYOUT_DEVICE=cpu
GLM_OCR_MAX_WORKERS=4
GLM_OCR_REQUEST_TIMEOUT_SECONDS=300
OCR_POLICY=auto

RAGLIGHT_ENABLED=true
LIGHTRAG_ENABLED=true
```

Los valores `sk-ejemplo-reemplazar` son placeholders y no se deben commitear.

## Ejecución local con Conda

Esta ruta sirve cuando quiero depurar desde el IDE. Como Docling y RAGLight tienen un conflicto real de `typer`, uso dos environments Conda separados. No creo `.venv` y no hago instalaciones manuales fuera de los archivos `environment.yml`. Para probar GLM-OCR desde el backend local puedo dejar levantado solo el servicio Docker `glm-ocr` en el puerto 8080; el `.env.example` apunta a `localhost:8080` y Compose sobrescribe esa URL cuando todo corre dentro de la red Docker.

### 1. Crear el entorno principal

Desde la raíz:

```bash
conda env create -f environment.yml
```

Si ya existe:

```bash
conda env update -n axiz-adaptive-rag-payments -f environment.yml --prune
```

Este entorno contiene backend, Docling, GLM-OCR, LightRAG y Node 22 para Angular.

### 2. Crear el entorno RAGLight

```bash
conda env create -f raglight_service/environment.yml
```

Si ya existe:

```bash
conda env update -n axiz-raglight-service -f raglight_service/environment.yml --prune
```

Este segundo environment contiene RAGLight 3.4.7 y su propio árbol de dependencias.

### 3. Levantar solo la infraestructura persistente

En una terminal:

```bash
docker compose -f infrastructure/docker-compose.yml up qdrant memgraph
```

### 4. Ejecutar RAGLight con Conda

En otra terminal, desde la raíz:

```bash
conda run -n axiz-raglight-service python -m uvicorn pe.axiz.raglight_service.main:app --host 0.0.0.0 --port 8010
```

### 5. Ejecutar backend con Conda

En otra terminal:

```bash
conda run -n axiz-adaptive-rag-payments python -m uvicorn pe.axiz.payment_knowledge.main:app --host 0.0.0.0 --port 8000
```

La configuración local por defecto usa `RAGLIGHT_SERVICE_URL=http://localhost:8010`.

### 6. Precargar el dataset con Conda

Con backend y RAGLight arriba:

```bash
conda run -n axiz-adaptive-rag-payments python datasets/seed.py --wait-seconds 900
```

`--wait-seconds` no limita la duración de la indexación. Si quiero imponer explícitamente un máximo de procesamiento, por ejemplo una hora, puedo usar:

```bash
conda run -n axiz-adaptive-rag-payments python datasets/seed.py --wait-seconds 900 --processing-timeout-seconds 3600
```

Con `--processing-timeout-seconds 0` (valor por defecto) la lectura queda sin límite y el proceso termina cuando el backend completa o devuelve un error real.

Al terminar deberían existir:

- colección principal `axiz_payment_chunks` en Qdrant;
- colección `axiz_payment_raglight`;
- nodos `Document`, `Chunk` y `Entity` en Memgraph;
- relaciones `CONTAINS`, `MENTIONS` y `CO_OCCURS`;
- working directory de LightRAG si está configurado.

### 7. Ejecutar Angular con Conda

Primero instalo los paquetes npm usando Node del entorno Conda principal:

```bash
conda run -n axiz-adaptive-rag-payments npm --prefix frontend install --no-audit --no-fund
```

Luego:

```bash
conda run -n axiz-adaptive-rag-payments npm --prefix frontend start
```

Abrir `http://localhost:4200`. En desarrollo Angular usa `frontend/proxy.conf.json`; en Docker Nginx hace el mismo proxy. El frontend consume `/api/v1` y no tiene `http://localhost:8000` hardcodeado.

La interfaz permite:

- precargar nuevamente el dataset;
- subir documentos;
- elegir `auto`, Native RAG, Hybrid RAG, RAGLight, GraphRAG o LightRAG;
- cambiar `top_k`;
- mostrar u ocultar trazabilidad y evidencia;
- ejecutar evaluación de los cinco motores;
- consultar relaciones de Memgraph desde el panel derecho.

Para generar el build Angular desde Conda:

```bash
conda run -n axiz-adaptive-rag-payments npm --prefix frontend run build
```

## Qué se inserta automáticamente al levantar Docker

El servicio `dataset-seed` usa `datasets/seed.py` y separa dos tiempos distintos. `--wait-seconds 900` controla únicamente cuánto se espera a que backend, Qdrant y Memgraph estén disponibles. Una vez que la infraestructura responde, la llamada a `POST /api/v1/datasets/seed` se ejecuta una sola vez y por defecto no tiene timeout de lectura (`--processing-timeout-seconds 0`), de modo que LightRAG pueda terminar una indexación larga sin ser abortado artificialmente a los 900 segundos. Los errores funcionales HTTP del seed siguen siendo terminales y no se reintenta toda la carga para evitar duplicar trabajo parcial.

La carga recorre `datasets/sample_documents` y ejecuta el mismo pipeline que una carga manual:

```text
documento
  -> Docling / GLM-OCR si corresponde
  -> Markdown canónico
  -> chunking + entidades
  -> Qdrant
  -> Memgraph
  -> RAGLight
  -> LightRAG cuando está configurado
```

Qdrant usa IDs deterministas para los chunks y Memgraph usa `MERGE`, por lo que volver a cargar el corpus no crea copias nuevas de esos elementos base. RAGLight usa IDs propios no deterministas, por eso su colección se reinicia antes del seed y se indexa el corpus completo una sola vez. RAGLight 3.4.7 fija `qdrant-client==1.17.0`, mientras que el backend principal usa `qdrant-client==1.19.1`. Por eso Qdrant Server se fija en `1.18.3`: es la versión más reciente de la rama 1.18 y queda a una versión menor de ambos clientes, evitando el conflicto del resolver y la advertencia de compatibilidad en runtime.

## Pruebas con curl

### Prueba 1: salud

```bash
curl http://localhost:8000/api/v1/health
```

Sirve para separar un problema de la API de un problema de Qdrant o Memgraph. El estado `ok` requiere ambos stores disponibles. `lightrag=false` es esperable si no se configuró un proveedor OpenAI-compatible.

### Prueba 2: cargar dataset

```bash
curl -X POST http://localhost:8000/api/v1/datasets/seed
```

Procesa los cuatro documentos de ejemplo. Esta llamada prueba la ruta completa de ingesta: parsing, chunking, embeddings, Qdrant, Memgraph y los adapters alternos habilitados.

### Prueba 3: Native RAG

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question":"¿Cómo evita la idempotencia cobros duplicados cuando hay timeout?",
    "strategy":"native_rag",
    "top_k":5,
    "include_trace":true,
    "include_evidence":true
  }'
```

Prueba el baseline vectorial puro.

### Prueba 4: Hybrid RAG con código exacto

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  --data-binary @infrastructure/requests/query-hybrid.json
```

Está pensada para comprobar que una consulta con códigos ISO 8583 aprovecha el ranking lexical además del vectorial.

### Prueba 5: Adaptive Routing

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  --data-binary @infrastructure/requests/query-auto.json
```

La pregunta incluida habla de componentes relacionados entre autorización y conciliación, por lo que el trace debería mostrar `reason=relaciones` y ejecutar `graphrag`.

### Prueba 6: RAGLight

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question":"¿Qué significa el código 05 en una autorización?",
    "strategy":"raglight",
    "top_k":5,
    "include_trace":true,
    "include_evidence":true
  }'
```

Fuerza el camino alterno RAGLight sobre su colección Qdrant. Esto permite compararlo con `hybrid_rag` usando la misma pregunta.

### Prueba 7: GraphRAG

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question":"¿Qué componentes están relacionados entre autorización y conciliación?",
    "strategy":"graphrag",
    "top_k":5,
    "include_trace":true,
    "include_evidence":true
  }'
```

Fuerza la fusión entre evidencia de Memgraph e Hybrid RAG.

### Prueba 8: explorar Memgraph

```bash
curl "http://localhost:8000/api/v1/graph/neighborhood/AUTORIZACION"
```

La búsqueda de entidad normaliza mayúsculas y acentos, por lo que `AUTORIZACION`, `autorización` y `Autorización` se comparan de forma equivalente contra los nombres de entidad existentes.

Devuelve las entidades que comparten chunks con la entidad consultada y el peso basado en cantidad de chunks compartidos.

### Prueba 9: LightRAG

Con `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL` y dimensión configurados:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question":"Relaciona autorización, adquirente y conciliación",
    "strategy":"lightrag",
    "top_k":5,
    "include_trace":true,
    "include_evidence":true
  }'
```

Sin proveedor configurado devuelve un error controlado indicando que LightRAG necesita esa configuración.

### Prueba 10: evaluación

```bash
curl -X POST http://localhost:8000/api/v1/evaluations/run \
  -H "Content-Type: application/json" \
  --data-binary @infrastructure/requests/evaluation.json
```

Ejecuta el mismo set contra Native RAG, Hybrid RAG, RAGLight, GraphRAG y LightRAG. Si un motor no está disponible, la corrida no se corta: sus casos quedan con `success=false` y la suma aparece en `failed_cases`. RAGLight y LightRAG exponen ahora el nombre real del documento recuperado (`authorization_codes.md`, `reconciliation.md`, etc.) en lugar de devolver únicamente `raglight` o `LightRAG`; de esta forma `Hit Rate` y `MRR` miden retrieval real y no el nombre del framework.

## Probar GLM-OCR

Incluyo un PDF y un JPG escaneados en `datasets/ocr_samples/`. La prueba recomendada fuerza GLM-OCR para que Docling no pueda resolver el documento antes:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/ingest?ocr=glm" \
  -F "file=@datasets/ocr_samples/documento_escaneado_prueba_ocr_pagos.pdf"
```

La respuesta debe incluir:

```json
{
  "processor": "glm-ocr",
  "ocr_used": true,
  "raglight_indexed": true,
  "lightrag_indexed": true
}
```

También puedo comparar el mismo archivo con Docling:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/ingest?ocr=docling" \
  -F "file=@datasets/ocr_samples/documento_escaneado_prueba_ocr_pagos.pdf"
```

`ocr=auto` mantiene el comportamiento de producción: imágenes van directo a GLM-OCR; los PDF digitales pasan primero por Docling y usan GLM-OCR solo si la extracción queda por debajo de `OCR_MIN_TEXT_CHARS`.

Para comprobar el modelo vLLM directamente:

```bash
curl http://localhost:8080/v1/models
```

Y el health del backend ahora debe mostrar `"glm_ocr": true`:

```bash
curl http://localhost:8000/api/v1/health
```

## Evaluación incluida

`datasets/evaluation/questions.json` define preguntas y fuentes esperadas. La evaluación está centrada en retrieval, no en calidad estilística del LLM. Por eso calcula:

- `hit_rate`: si alguna fuente esperada aparece en Top K;
- `mrr`: qué tan arriba aparece la primera fuente esperada;
- `avg_latency_ms`: tiempo promedio del retrieval;
- `successful_cases` y `failed_cases`: permite comparar sin ocultar que un motor estaba deshabilitado o mal configurado.

Para RAGLight, cuando el framework no entrega `source` en su metadata, el adapter atribuye cada resultado comparando el texto recuperado contra los Markdown canónicos y deja el método/score en `metadata.source_attribution`. Para LightRAG se interpreta `Reference Document List` y los `reference_id` del contexto generado por el SDK, conservando el orden de ranking que devolvió LightRAG. Si el formato cambia, se conserva un fallback explícito en vez de inventar una fuente.

Para una evaluación más seria se puede ampliar el dataset sin cambiar el código del servicio.

## Pruebas de código

Todas las pruebas locales se ejecutan dentro del entorno Conda.

Backend completo:

```bash
conda run -n axiz-adaptive-rag-payments python scripts/verify.py
```

Pruebas unitarias del backend:

```bash
conda run -n axiz-adaptive-rag-payments python -m pytest -q backend/tests
```

Pruebas del servicio RAGLight:

```bash
conda run -n axiz-raglight-service python -m pytest -q raglight_service/tests
```

Chequeo de estilo:

```bash
conda run -n axiz-adaptive-rag-payments ruff check backend/src backend/tests datasets
conda run -n axiz-raglight-service ruff check raglight_service/src raglight_service/tests
```

Frontend:

```bash
conda run -n axiz-adaptive-rag-payments npm --prefix frontend run build
```

## Estado de validación de esta entrega

Antes de empaquetar esta versión validé:

- `compileall` del backend, dataset y servicio RAGLight;
- `28 passed` en `backend/tests`, incluyendo política OCR `auto/glm/docling` y URL self-hosted;
- `8 passed` en `raglight_service/tests`;
- regresión de atribución de fuentes RAGLight/LightRAG;
- regresión de generación OpenAI-compatible sin `temperature` forzado;
- normalización de mayúsculas/acentos para `graph/neighborhood`;
- sintaxis YAML/TOML y resolución de servicios del Compose;
- construcción de wheels del backend y RAGLight sin instalar dependencias;
- integridad del ZIP final.

El entorno usado para preparar el ZIP no tiene runtime Docker/NVIDIA, por lo que la ejecución real del nuevo contenedor vLLM debe validarse en la máquina destino. El Compose sigue la receta oficial de GLM-OCR/vLLM: modelo `zai-org/GLM-OCR`, vLLM 0.19.0, Transformers 5.3.0 y API OpenAI-compatible. En la v15 se corrigió específicamente el build de `Dockerfile.glm-ocr`: la imagen oficial no garantiza el comando `python`, pero sí `python3`, de modo que la capa usa `python3 -m pip` y valida en build las versiones efectivas de Python, Transformers y vLLM.

## Decisiones de alcance

- Qdrant y Memgraph son los únicos stores de infraestructura. GLM-OCR y RAGLight son servicios de aplicación especializados que se ejecutan en contenedores separados por compatibilidad y aislamiento.
- RAGLight reutiliza Qdrant pero corre en un servicio Python separado y usa una colección distinta para que la comparación no mezcle índices ni árboles de dependencias.
- GraphRAG es una implementación propia sobre Memgraph y Hybrid RAG; LightRAG se mantiene como flujo alterno para comparación.
- La generación se desacopla del retrieval. Esto permite medir retrieval sin que una respuesta de LLM cambie el resultado de Hit Rate o MRR.
- El modo `auto` no selecciona RAGLight/LightRAG para evitar que el comportamiento base dependa de frameworks o credenciales externas. Esos motores se pueden forzar desde API/UI y se incluyen en evaluación.
- No se agregó un broker, una base relacional o una base documental porque no hay un requisito del caso que lo justifique.
- GLM-OCR usa un contenedor vLLM dedicado con GPU y persistencia de caché; el pipeline/layout sigue en el backend y corre en CPU.

## Optimización del build Docker para dependencias de IA

Docling, el pipeline self-hosted de GLM-OCR y RAGLight terminan usando PyTorch. Si se deja que `pip` resuelva PyTorch desde el índice general de PyPI en Linux, puede descargar además paquetes CUDA/NVIDIA de varios gigabytes aunque esta PoC se ejecute en CPU. Eso hace que el primer `docker compose up --build` tarde demasiado.

Backend y RAGLight se construyen desde `infrastructure/Dockerfile.ai` con una etapa CPU compartida que instala explícitamente PyTorch CPU. GLM-OCR usa `infrastructure/Dockerfile.glm-ocr`, basado en la imagen oficial GPU de vLLM, y se mantiene separado para no contaminar el backend con CUDA. BuildKit mantiene caché de `pip`, y Docker reutiliza las capas de la imagen vLLM ya descargadas; por eso, después de una descarga inicial completa, una corrección en la capa final de `Dockerfile.glm-ocr` no debería volver a descargar los varios GB de la imagen base salvo que se haya limpiado la caché local.

No se usa Conda dentro de Docker. Conda queda únicamente para trabajar localmente desde el IDE.

El primer build seguirá siendo más pesado que un servicio web normal porque Docling, Transformers y Sentence Transformers son dependencias de IA, pero no debería descargar el stack CUDA completo dos veces.

Para ver el detalle de lo que Docker está descargando o instalando:

```bash
BUILDKIT_PROGRESS=plain docker compose -f infrastructure/docker-compose.yml build backend raglight-service glm-ocr
```

Después se levanta todo normalmente:

```bash
docker compose -f infrastructure/docker-compose.yml up --build
```



### Error `python: not found` al construir GLM-OCR

La imagen `vllm/vllm-openai:v0.19.0-ubuntu2404` no garantiza el alias `python`; el intérprete disponible es `python3`. Si aparece:

```text
/bin/sh: 1: python: not found
```

la v15 ya lo corrige usando:

```dockerfile
RUN python3 -m pip install --upgrade "transformers==5.3.0"
```

Además, la misma capa hace un import de `transformers` y `vllm` y muestra sus versiones durante el build. Si la imagen base ya terminó de descargarse en un intento anterior, Docker debería reutilizarla.

### Compatibilidad Qdrant y RAGLight

No se debe agregar manualmente `qdrant-client==1.19.1` al servicio RAGLight. RAGLight 3.4.7 declara `qdrant-client==1.17.0` en su extra `qdrant`; forzarlo a 1.19.1 hace que `pip` termine con `ResolutionImpossible`. La PoC usa Qdrant Server 1.18.3 para mantener compatibilidad simultánea con el cliente 1.17.0 de RAGLight y el 1.19.1 del backend principal.

