import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, computed, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

interface ContextItem {
  id: string;
  document_id: string;
  source: string;
  title: string;
  text: string;
  score: number;
  strategy: string;
  entities: string[];
  metadata: Record<string, unknown>;
}

interface QueryResponse {
  answer: string;
  requested_strategy: string;
  executed_strategy: string;
  contexts: ContextItem[];
  trace: Record<string, unknown>;
  timings_ms: Record<string, number>;
}

interface IngestResponse {
  document_id: string;
  source: string;
  processor: string;
  chunks: number;
  indexed: boolean;
  idempotency_key: string;
  replaced_existing: boolean;
  timings_ms: Record<string, number>;
}

interface DocumentSummary {
  document_id: string;
  source: string;
  title: string;
  processor: string;
  qdrant_chunks: number;
  memgraph_chunks: number;
  canonical_files: number;
  entities: string[];
  consistent: boolean;
}

interface JobResponse {
  job_id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed';
  stage: string;
  progress: number;
  source: string;
  created_at: string;
  updated_at: string;
  result?: IngestResponse;
  error?: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  payload?: QueryResponse;
}

interface RankingComparison {
  strategy: string;
  raw_mrr: number;
  reranked_mrr: number;
  delta_mrr: number;
  raw_ndcg_at_k: number;
  reranked_ndcg_at_k: number;
  delta_ndcg_at_k: number;
  hit_rate: number;
  recall_at_k: number;
  avg_latency_ms: number;
  avg_duplicates_removed: number;
  routing_accuracy?: number;
  successful_cases: number;
  failed_cases: number;
}

interface EvaluationResponse {
  dataset_cases: number;
  summary: Record<string, Record<string, number>>;
  ranking_comparison: RankingComparison[];
  category_summary: Record<string, Record<string, Record<string, number>>>;
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

interface GraphNode {
  id: string;
  mentions: number;
}

interface GraphNodeView extends GraphNode {
  x: number;
  y: number;
}

type Workspace = 'chat' | 'documents' | 'graph' | 'evaluations';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class AppComponent {
  readonly Math = Math;
  private readonly apiBase = '/api/v1';

  readonly strategies = [
    ['auto', 'Adaptive Routing'],
    ['native_rag', 'Native RAG'],
    ['hybrid_rag', 'Hybrid RAG'],
    ['raglight', 'RAGLight'],
    ['graphrag', 'GraphRAG'],
    ['lightrag', 'LightRAG']
  ];

  readonly examples = [
    '¿Qué significa el código 05 y qué debería revisar operaciones?',
    '¿Qué componentes están relacionados entre autorización y conciliación?',
    '¿Cómo evita la idempotencia cobros duplicados cuando hay timeout?',
    '¿Para qué sirve la tokenización respecto al PAN?'
  ];

  workspace = signal<Workspace>('chat');
  question = '';
  strategy = 'auto';
  topK = 5;
  showTrace = true;
  showEvidence = true;
  rerank = true;
  deduplicate = true;
  loading = signal(false);
  apiStatus = signal<'checking' | 'online' | 'offline'>('checking');
  messages = signal<ChatMessage[]>([]);
  evaluation = signal<EvaluationResponse | null>(null);
  documents = signal<DocumentSummary[]>([]);
  jobs = signal<JobResponse[]>([]);
  activeJob = signal<JobResponse | null>(null);
  selectedFile = signal<File | null>(null);
  uploadExternal = false;
  graphEntity = 'AUTORIZACIÓN';
  graphEdges = signal<GraphEdge[]>([]);
  graphNodes = signal<GraphNodeView[]>([]);
  graphOverviewEdges = signal<GraphEdge[]>([]);
  notice = signal('');

  readonly title = computed(() => {
    if (this.workspace() === 'documents') return 'Administración de documentos';
    if (this.workspace() === 'graph') return 'Explorador de conocimiento';
    if (this.workspace() === 'evaluations') return 'Evals y ranking de estrategias';
    return this.messages().find((message) => message.role === 'user')?.content || 'Nueva conversación';
  });

  readonly rankingRows = computed(() => this.evaluation()?.ranking_comparison || []);
  readonly categoryEntries = computed(() => Object.entries(this.evaluation()?.category_summary || {}));

  constructor(private readonly http: HttpClient) {
    this.checkHealth();
    this.refreshDocuments();
    this.refreshJobs();
  }

  switchWorkspace(workspace: Workspace): void {
    this.workspace.set(workspace);
    this.notice.set('');
    if (workspace === 'documents') {
      this.refreshDocuments();
      this.refreshJobs();
    }
    if (workspace === 'graph') this.loadGraphOverview();
  }

  checkHealth(): void {
    this.http.get(`${this.apiBase}/health`).subscribe({
      next: () => this.apiStatus.set('online'),
      error: () => this.apiStatus.set('offline')
    });
  }

  useExample(example: string): void {
    this.workspace.set('chat');
    this.question = example;
  }

  newConversation(): void {
    this.workspace.set('chat');
    this.messages.set([]);
    this.question = '';
  }

  onComposerKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }

  send(): void {
    const question = this.question.trim();
    if (!question || this.loading()) return;
    this.messages.update((items) => [...items, { role: 'user', content: question }]);
    this.question = '';
    this.loading.set(true);
    this.http.post<QueryResponse>(`${this.apiBase}/query`, {
      question,
      strategy: this.strategy,
      top_k: this.topK,
      include_trace: this.showTrace,
      include_evidence: this.showEvidence,
      rerank: this.rerank,
      deduplicate: this.deduplicate
    }).subscribe({
      next: (payload) => {
        this.messages.update((items) => [...items, { role: 'assistant', content: payload.answer, payload }]);
        this.loading.set(false);
      },
      error: (error) => {
        const detail = error?.error?.detail || 'No se pudo completar la consulta.';
        this.messages.update((items) => [...items, { role: 'assistant', content: detail }]);
        this.loading.set(false);
      }
    });
  }

  seed(): void {
    if (this.loading()) return;
    this.loading.set(true);
    this.http.post(`${this.apiBase}/datasets/seed`, {}).subscribe({
      next: () => {
        this.loading.set(false);
        this.notice.set('Dataset base cargado en los motores de recuperación.');
        this.refreshDocuments();
      },
      error: (error) => {
        this.loading.set(false);
        this.notice.set(error?.error?.detail || 'No se pudo cargar el dataset.');
      }
    });
  }

  runEvaluation(): void {
    if (this.loading()) return;
    this.loading.set(true);
    this.workspace.set('evaluations');
    this.http.post<EvaluationResponse>(`${this.apiBase}/evaluations/run`, {
      strategies: ['native_rag', 'hybrid_rag', 'raglight', 'graphrag', 'lightrag', 'auto'],
      top_k: this.topK,
      compare_reranking: true
    }).subscribe({
      next: (result) => {
        this.evaluation.set(result);
        this.loading.set(false);
      },
      error: (error) => {
        this.notice.set(error?.error?.detail || 'No se pudo ejecutar la evaluación.');
        this.loading.set(false);
      }
    });
  }

  onFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.selectedFile.set(input.files?.[0] || null);
  }

  uploadAsync(): void {
    const file = this.selectedFile();
    if (!file || this.loading()) return;
    const form = new FormData();
    form.append('file', file);
    this.loading.set(true);
    const external = this.uploadExternal ? 'true' : 'false';
    this.http.post<JobResponse>(`${this.apiBase}/documents/jobs?index_external=${external}`, form).subscribe({
      next: (job) => {
        this.activeJob.set(job);
        this.loading.set(false);
        this.notice.set(`Job ${job.job_id.slice(0, 8)} creado para ${job.source}.`);
        this.refreshJobs();
        this.pollJob(job.job_id);
      },
      error: (error) => {
        this.loading.set(false);
        this.notice.set(error?.error?.detail || 'No se pudo crear el job de ingesta.');
      }
    });
  }

  private pollJob(jobId: string): void {
    this.http.get<JobResponse>(`${this.apiBase}/jobs/${jobId}`).subscribe({
      next: (job) => {
        this.activeJob.set(job);
        this.refreshJobs();
        if (job.status === 'queued' || job.status === 'running') {
          setTimeout(() => this.pollJob(jobId), 2000);
          return;
        }
        if (job.status === 'succeeded') {
          this.notice.set(`${job.source} indexado correctamente.`);
          this.selectedFile.set(null);
          this.refreshDocuments();
        } else {
          this.notice.set(job.error || 'El job de ingesta falló.');
        }
      },
      error: () => this.notice.set('No se pudo consultar el estado del job.')
    });
  }

  refreshJobs(): void {
    this.http.get<JobResponse[]>(`${this.apiBase}/jobs?limit=20`).subscribe({
      next: (jobs) => this.jobs.set(jobs),
      error: () => this.jobs.set([])
    });
  }

  refreshDocuments(): void {
    this.http.get<DocumentSummary[]>(`${this.apiBase}/documents`).subscribe({
      next: (documents) => this.documents.set(documents),
      error: () => this.documents.set([])
    });
  }

  deleteDocument(document: DocumentSummary): void {
    if (!confirm(`Eliminar ${document.source} de Qdrant, Memgraph y corpus canónico?`)) return;
    this.http.delete(`${this.apiBase}/documents/${document.document_id}`).subscribe({
      next: () => {
        this.notice.set(`${document.source} eliminado.`);
        this.refreshDocuments();
        this.loadGraphOverview();
      },
      error: (error) => this.notice.set(error?.error?.detail || 'No se pudo eliminar el documento.')
    });
  }

  reindexDocument(document: DocumentSummary): void {
    this.http.post<IngestResponse>(`${this.apiBase}/documents/${document.document_id}/reindex`, {}).subscribe({
      next: () => {
        this.notice.set(`${document.source} reindexado desde el canónico.`);
        this.refreshDocuments();
      },
      error: (error) => this.notice.set(error?.error?.detail || 'No se pudo reindexar el documento.')
    });
  }

  loadGraph(): void {
    const entity = this.graphEntity.trim();
    if (!entity) return;
    this.http.get<{ edges: GraphEdge[] }>(`${this.apiBase}/graph/neighborhood/${encodeURIComponent(entity)}`).subscribe({
      next: (result) => this.graphEdges.set(result.edges || []),
      error: () => this.graphEdges.set([])
    });
  }

  loadGraphOverview(): void {
    this.http.get<{ nodes: GraphNode[]; edges: GraphEdge[] }>(`${this.apiBase}/graph/overview?limit=60`).subscribe({
      next: (result) => {
        this.graphOverviewEdges.set(result.edges || []);
        this.graphNodes.set(this.layoutNodes(result.nodes || []));
      },
      error: () => {
        this.graphNodes.set([]);
        this.graphOverviewEdges.set([]);
      }
    });
  }

  private layoutNodes(nodes: GraphNode[]): GraphNodeView[] {
    const width = 760;
    const height = 420;
    const radius = Math.min(width, height) * 0.38;
    return nodes.slice(0, 24).map((node, index, all) => {
      const angle = (2 * Math.PI * index) / Math.max(all.length, 1) - Math.PI / 2;
      return {
        ...node,
        x: width / 2 + radius * Math.cos(angle),
        y: height / 2 + radius * Math.sin(angle)
      };
    });
  }

  nodeById(id: string): GraphNodeView | undefined {
    return this.graphNodes().find((node) => node.id === id);
  }

  traceEntries(payload: QueryResponse): [string, unknown][] {
    return Object.entries(payload.trace || {});
  }

  summaryEntries(): [string, Record<string, number>][] {
    return Object.entries(this.evaluation()?.summary || {});
  }

  strategyLabel(value: string): string {
    return this.strategies.find((entry) => entry[0] === value)?.[1] || value;
  }

  deltaLabel(value: number): string {
    if (value > 0.0001) return `+${value.toFixed(3)}`;
    return value.toFixed(3);
  }
}
