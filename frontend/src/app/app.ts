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

interface ExecutionStep {
  label: string;
  status: 'active' | 'done' | 'error';
  detail?: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  payload?: QueryResponse;
  streaming?: boolean;
  steps?: ExecutionStep[];
}

interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
}

interface ConversationGroup {
  label: string;
  conversations: Conversation[];
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
  private readonly conversationsKey = 'axiz-adaptive-rag-conversations-v1';
  private followStream = true;

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
  conversations = signal<Conversation[]>([]);
  activeConversationId = signal('');
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
    return this.activeConversation()?.title || 'Nueva conversación';
  });

  readonly rankingRows = computed(() => this.evaluation()?.ranking_comparison || []);
  readonly categoryEntries = computed(() => Object.entries(this.evaluation()?.category_summary || {}));

  constructor(private readonly http: HttpClient) {
    this.restoreConversations();
    this.checkHealth();
    this.refreshDocuments();
    this.refreshJobs();
  }

  activeConversation(): Conversation | undefined {
    return this.conversations().find((item) => item.id === this.activeConversationId());
  }

  conversationGroups(): ConversationGroup[] {
    const now = new Date();
    const buckets = new Map<string, Conversation[]>();
    const order = ['Hoy', 'Ayer', 'Últimos 7 días', 'Últimos 30 días', 'Anteriores'];
    for (const conversation of [...this.conversations()].sort(
      (a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt)
    )) {
      const label = this.conversationGroupLabel(new Date(conversation.updatedAt), now);
      const group = buckets.get(label) || [];
      group.push(conversation);
      buckets.set(label, group);
    }
    return order.filter((label) => buckets.has(label)).map((label) => ({
      label,
      conversations: buckets.get(label) || []
    }));
  }

  switchWorkspace(workspace: Workspace): void {
    this.workspace.set(workspace);
    this.notice.set('');
    if (workspace === 'documents') {
      this.refreshDocuments();
      this.refreshJobs();
    }
    if (workspace === 'graph') this.loadGraphOverview();
    if (workspace === 'chat') setTimeout(() => this.focusComposer(), 0);
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
    setTimeout(() => this.focusComposer(), 0);
  }

  newConversation(): void {
    if (this.loading()) return;
    const conversation = this.createConversation();
    this.conversations.update((items) => [conversation, ...items]);
    this.activeConversationId.set(conversation.id);
    this.messages.set([]);
    this.workspace.set('chat');
    this.question = '';
    this.persistConversations();
    setTimeout(() => this.focusComposer(), 0);
  }

  openConversation(id: string): void {
    if (this.loading()) return;
    const conversation = this.conversations().find((item) => item.id === id);
    if (!conversation) return;
    this.activeConversationId.set(id);
    this.messages.set(conversation.messages.map((message) => ({ ...message, streaming: false })));
    this.workspace.set('chat');
    this.notice.set('');
    this.followStream = true;
    setTimeout(() => {
      this.scrollToBottom(true);
      this.focusComposer();
    }, 0);
  }

  deleteConversation(event: Event, id: string): void {
    event.stopPropagation();
    if (this.loading()) return;
    this.conversations.update((items) => items.filter((item) => item.id !== id));
    if (this.activeConversationId() === id) {
      const next = this.conversations()[0];
      if (next) {
        this.activeConversationId.set(next.id);
        this.messages.set(next.messages);
      } else {
        const conversation = this.createConversation();
        this.conversations.set([conversation]);
        this.activeConversationId.set(conversation.id);
        this.messages.set([]);
      }
    }
    this.persistConversations();
  }

  onComposerKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void this.send();
    }
  }

  onChatScroll(): void {
    const panel = document.querySelector<HTMLElement>('.chat-panel');
    if (!panel) return;
    const distance = panel.scrollHeight - panel.scrollTop - panel.clientHeight;
    this.followStream = distance < 90;
  }

  async send(): Promise<void> {
    const question = this.question.trim();
    if (!question || this.loading()) return;

    const userMessage: ChatMessage = {
      id: this.uuid(),
      role: 'user',
      content: question
    };
    const assistantId = this.uuid();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      streaming: true,
      steps: [{ label: 'Conectando con el pipeline RAG', status: 'active' }]
    };

    this.messages.update((items) => [...items, userMessage, assistantMessage]);
    this.question = '';
    this.loading.set(true);
    this.followStream = true;
    this.updateConversationTitle(question);
    this.persistActiveConversation();
    requestAnimationFrame(() => this.scrollToBottom(true));

    let completed = false;
    try {
      const response = await fetch(`${this.apiBase}/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
          'Cache-Control': 'no-cache'
        },
        body: JSON.stringify({
          question,
          strategy: this.strategy,
          top_k: this.topK,
          include_trace: this.showTrace,
          include_evidence: this.showEvidence,
          rerank: this.rerank,
          deduplicate: this.deduplicate
        })
      });
      if (!response.ok || !response.body) {
        const detail = await response.text();
        throw new Error(detail || `HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
        let boundary = buffer.indexOf('\n\n');
        while (boundary >= 0) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          completed = this.handleSseBlock(block, assistantId) || completed;
          boundary = buffer.indexOf('\n\n');
        }
      }
      const tail = buffer.trim();
      if (tail) completed = this.handleSseBlock(tail, assistantId) || completed;
      if (!completed) {
        this.finishStreamingMessage(assistantId);
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'No se pudo completar la consulta.';
      this.failStreamingMessage(assistantId, detail);
    } finally {
      this.loading.set(false);
      this.persistActiveConversation();
      setTimeout(() => {
        this.scrollToBottom();
        this.focusComposer();
      }, 0);
    }
  }

  private handleSseBlock(block: string, assistantId: string): boolean {
    if (!block.trim() || block.startsWith(':')) return false;
    let eventName = 'message';
    const dataLines: string[] = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) eventName = line.slice(6).trim();
      if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return false;
    let data: Record<string, unknown> = {};
    try {
      data = JSON.parse(dataLines.join('\n')) as Record<string, unknown>;
    } catch {
      return false;
    }

    if (eventName === 'stage') {
      this.pushExecutionStep(assistantId, String(data['message'] || 'Procesando'));
    } else if (eventName === 'retrieval') {
      const contexts = Number(data['returned_contexts'] || 0);
      const retrievalMs = Number(data['retrieval_ms'] || 0);
      const strategy = String(data['executed_strategy'] || 'RAG');
      this.pushExecutionStep(
        assistantId,
        'Evidencia recuperada y ranking aplicado',
        `${strategy} · ${contexts} contextos · ${retrievalMs.toFixed(0)} ms`
      );
    } else if (eventName === 'delta') {
      const delta = String(data['delta'] || '');
      if (delta) this.appendAssistantDelta(assistantId, delta);
    } else if (eventName === 'complete') {
      const payload = data as unknown as QueryResponse;
      this.completeStreamingMessage(assistantId, payload);
      return true;
    } else if (eventName === 'error') {
      throw new Error(String(data['detail'] || data['message'] || 'Error SSE'));
    }
    return false;
  }

  private appendAssistantDelta(id: string, delta: string): void {
    const shouldFollow = this.followStream;
    this.messages.update((items) => items.map((message) =>
      message.id === id ? { ...message, content: message.content + delta } : message
    ));
    if (shouldFollow) requestAnimationFrame(() => this.scrollToBottom());
  }

  private pushExecutionStep(id: string, label: string, detail?: string): void {
    this.messages.update((items) => items.map((message) => {
      if (message.id !== id) return message;
      const previous = (message.steps || []).map((step) =>
        step.status === 'active' ? { ...step, status: 'done' as const } : step
      );
      return {
        ...message,
        steps: [...previous, { label, detail, status: 'active' }]
      };
    }));
    if (this.followStream) requestAnimationFrame(() => this.scrollToBottom());
  }

  private completeStreamingMessage(id: string, payload: QueryResponse): void {
    this.messages.update((items) => items.map((message) => {
      if (message.id !== id) return message;
      const steps = (message.steps || []).map((step) => ({ ...step, status: 'done' as const }));
      return {
        ...message,
        content: payload.answer || message.content,
        payload,
        streaming: false,
        steps: [...steps, { label: 'Respuesta completada', status: 'done' as const }]
      };
    }));
    this.persistActiveConversation();
  }

  private finishStreamingMessage(id: string): void {
    this.messages.update((items) => items.map((message) => {
      if (message.id !== id) return message;
      const steps = (message.steps || []).map((step) => ({ ...step, status: 'done' as const }));
      return { ...message, streaming: false, steps };
    }));
  }

  private failStreamingMessage(id: string, detail: string): void {
    this.messages.update((items) => items.map((message) => {
      if (message.id !== id) return message;
      const steps = (message.steps || []).map((step) =>
        step.status === 'active' ? { ...step, status: 'error' as const } : step
      );
      return {
        ...message,
        content: message.content || `No se pudo completar la consulta: ${detail}`,
        streaming: false,
        steps
      };
    }));
  }

  private scrollToBottom(force = false): void {
    if (!force && !this.followStream) return;
    const panel = document.querySelector<HTMLElement>('.chat-panel');
    if (!panel) return;
    panel.scrollTop = panel.scrollHeight;
  }

  private focusComposer(): void {
    document.querySelector<HTMLTextAreaElement>('#chat-composer-input')?.focus();
  }

  private restoreConversations(): void {
    let restored: Conversation[] = [];
    try {
      const raw = localStorage.getItem(this.conversationsKey);
      if (raw) restored = JSON.parse(raw) as Conversation[];
    } catch {
      restored = [];
    }
    restored = restored.filter((item) => item?.id && Array.isArray(item.messages));
    if (!restored.length) restored = [this.createConversation()];
    this.conversations.set(restored);
    this.activeConversationId.set(restored[0].id);
    this.messages.set(restored[0].messages.map((message) => ({ ...message, streaming: false })));
  }

  private createConversation(): Conversation {
    const now = new Date().toISOString();
    return {
      id: this.uuid(),
      title: 'Nueva conversación',
      createdAt: now,
      updatedAt: now,
      messages: []
    };
  }

  private persistActiveConversation(): void {
    const id = this.activeConversationId();
    const now = new Date().toISOString();
    this.conversations.update((items) => items.map((conversation) =>
      conversation.id === id
        ? { ...conversation, messages: this.messages(), updatedAt: now }
        : conversation
    ));
    this.persistConversations();
  }

  private persistConversations(): void {
    try {
      localStorage.setItem(this.conversationsKey, JSON.stringify(this.conversations()));
    } catch {
      // La conversación sigue operativa aunque el navegador bloquee localStorage.
    }
  }

  private updateConversationTitle(question: string): void {
    const id = this.activeConversationId();
    this.conversations.update((items) => items.map((conversation) => {
      if (conversation.id !== id || conversation.title !== 'Nueva conversación') return conversation;
      const normalized = question.replace(/\s+/g, ' ').trim();
      const title = normalized.length <= 46 ? normalized : `${normalized.slice(0, 45).trim()}…`;
      return { ...conversation, title };
    }));
  }

  private conversationGroupLabel(updated: Date, now: Date): string {
    const oneDay = 24 * 60 * 60 * 1000;
    const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const startUpdated = new Date(
      updated.getFullYear(), updated.getMonth(), updated.getDate()
    ).getTime();
    const dayDiff = Math.round((startToday - startUpdated) / oneDay);
    if (dayDiff <= 0) return 'Hoy';
    if (dayDiff === 1) return 'Ayer';
    if (dayDiff <= 7) return 'Últimos 7 días';
    if (dayDiff <= 30) return 'Últimos 30 días';
    return 'Anteriores';
  }

  private uuid(): string {
    return typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
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
        const detail = error?.error?.detail;
        const status = Number(error?.status || 0);
        if (detail) {
          this.notice.set(detail);
        } else if (status === 504) {
          this.notice.set('La evaluación excedió el timeout del proxy HTTP (504).');
        } else if (status > 0) {
          this.notice.set(`No se pudo ejecutar la evaluación (HTTP ${status}).`);
        } else {
          this.notice.set('No se pudo ejecutar la evaluación.');
        }
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
