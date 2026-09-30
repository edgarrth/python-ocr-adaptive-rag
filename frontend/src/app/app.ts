import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, computed, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

interface ContextItem {
  id: string;
  source: string;
  title: string;
  text: string;
  score: number;
  strategy: string;
  entities: string[];
}

interface QueryResponse {
  answer: string;
  requested_strategy: string;
  executed_strategy: string;
  contexts: ContextItem[];
  trace: Record<string, unknown>;
  timings_ms: Record<string, number>;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  payload?: QueryResponse;
}

interface EvaluationResponse {
  summary: Record<string, Record<string, number>>;
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class AppComponent {
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

  question = '';
  strategy = 'auto';
  topK = 5;
  showTrace = true;
  showEvidence = true;
  loading = signal(false);
  apiStatus = signal<'checking' | 'online' | 'offline'>('checking');
  messages = signal<ChatMessage[]>([]);
  evaluation = signal<EvaluationResponse | null>(null);
  selectedFile = signal<File | null>(null);
  graphEntity = 'AUTORIZACIÓN';
  graphEdges = signal<GraphEdge[]>([]);
  readonly title = computed(() => this.messages().find((message) => message.role === 'user')?.content || 'Nueva conversación');

  constructor(private readonly http: HttpClient) {
    this.checkHealth();
  }

  checkHealth(): void {
    this.http.get(`${this.apiBase}/health`).subscribe({
      next: () => this.apiStatus.set('online'),
      error: () => this.apiStatus.set('offline')
    });
  }

  useExample(example: string): void {
    this.question = example;
  }

  newConversation(): void {
    this.messages.set([]);
    this.question = '';
    this.evaluation.set(null);
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
      include_evidence: this.showEvidence
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
        this.messages.update((items) => [...items, { role: 'assistant', content: 'Dataset de payment processing cargado en Qdrant, Memgraph y flujos habilitados.' }]);
      },
      error: (error) => {
        this.loading.set(false);
        this.messages.update((items) => [...items, { role: 'assistant', content: error?.error?.detail || 'No se pudo cargar el dataset.' }]);
      }
    });
  }

  runEvaluation(): void {
    if (this.loading()) return;
    this.loading.set(true);
    this.http.post<EvaluationResponse>(`${this.apiBase}/evaluations/run`, {
      strategies: ['native_rag', 'hybrid_rag', 'raglight', 'graphrag', 'lightrag'],
      top_k: this.topK
    }).subscribe({
      next: (result) => {
        this.evaluation.set(result);
        this.loading.set(false);
      },
      error: () => this.loading.set(false)
    });
  }

  onFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.selectedFile.set(input.files?.[0] || null);
  }

  upload(): void {
    const file = this.selectedFile();
    if (!file || this.loading()) return;
    const form = new FormData();
    form.append('file', file);
    this.loading.set(true);
    this.http.post(`${this.apiBase}/documents/ingest`, form).subscribe({
      next: (result: any) => {
        this.loading.set(false);
        this.messages.update((items) => [...items, { role: 'assistant', content: `Documento ${result.source} procesado con ${result.processor}; ${result.chunks} chunks indexados.` }]);
      },
      error: (error) => {
        this.loading.set(false);
        this.messages.update((items) => [...items, { role: 'assistant', content: error?.error?.detail || 'No se pudo procesar el documento.' }]);
      }
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

  traceEntries(payload: QueryResponse): [string, unknown][] {
    return Object.entries(payload.trace || {});
  }

  summaryEntries(): [string, Record<string, number>][] {
    return Object.entries(this.evaluation()?.summary || {});
  }
}
