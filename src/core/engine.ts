import type { WebLLMConfig, ModelLoadOptions, ModelHandle, EventHandler } from './types.js';
import { MemoryPool } from './memory-pool.js';
import { Scheduler } from './scheduler.js';
import { PipelineCache } from './pipeline-cache.js';

export class WebLLM {
  private config: WebLLMConfig;
  private memoryPool: MemoryPool;
  private scheduler: Scheduler;
  private pipelineCache: PipelineCache;
  private models = new Map<string, ModelHandle>();
  private eventHandlers = new Map<string, Set<EventHandler>>();

  private constructor(config: WebLLMConfig) {
    this.config = config;
    this.memoryPool = new MemoryPool(config.memoryBudget);
    this.scheduler = new Scheduler({ frameBudgetMs: config.frameBudgetMs ?? 8 });
    this.pipelineCache = new PipelineCache(config.cacheDir ?? 'webllm-cache');
  }

  static async init(config: WebLLMConfig): Promise<WebLLM> {
    return new WebLLM(config);
  }

  async loadModel(name: string, options: ModelLoadOptions): Promise<ModelHandle> {
    const id = `model-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const handle: ModelHandle = { id, name, priority: options.priority, lightweight: options.lightweight ?? false };
    this.models.set(id, handle);
    return handle;
  }

  async unloadModel(id: string): Promise<void> { this.models.delete(id); }
  getMemoryPool(): MemoryPool { return this.memoryPool; }
  getScheduler(): Scheduler { return this.scheduler; }

  on(event: string, handler: EventHandler): void {
    if (!this.eventHandlers.has(event)) this.eventHandlers.set(event, new Set());
    this.eventHandlers.get(event)!.add(handler);
  }

  off(event: string, handler: EventHandler): void { this.eventHandlers.get(event)?.delete(handler); }

  async shutdown(): Promise<void> {
    this.models.clear();
    this.memoryPool.reset();
    this.scheduler.clear();
    this.eventHandlers.clear();
  }
}
