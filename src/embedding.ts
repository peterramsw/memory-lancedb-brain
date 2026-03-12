/**
 * Embedding provider with OpenAI-compatible HTTP API support
 */

export interface EmbeddingConfig {
  apiKey: string;
  model: string;
  baseURL: string;
  dimensions?: number;
}

export interface EmbeddingResult {
  embedding: number[];
}

export interface EmbeddingResponse {
  data: EmbeddingResult[];
  usage?: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

export class Embedder {
  private config: EmbeddingConfig;

  constructor(config: EmbeddingConfig) {
    this.config = config;
  }

  /**
   * Create an embedder instance from config
   */
  static create(config: EmbeddingConfig): Embedder {
    return new Embedder(config);
  }

  /**
   * Default configuration
   */
  static defaultConfig(): EmbeddingConfig {
    return {
      apiKey: "local",
      model: "vllm/Forturne/Qwen3-Embedding-4B-NVFP4",
      baseURL: "http://127.0.0.1:32080/v1",
      dimensions: 2560,
    };
  }

  /**
   * Generate embedding for a single text
   */
  async embed(text: string): Promise<number[]> {
    const makeBody = (includeDimensions: boolean) => ({
      input: text,
      model: this.config.model,
      ...(includeDimensions && this.config.dimensions ? { dimensions: this.config.dimensions } : {}),
    });

    let response = await fetch(`${this.config.baseURL}/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` }),
      },
      body: JSON.stringify(makeBody(true)),
    });

    if (!response.ok && response.status === 400 && this.config.dimensions) {
      response = await fetch(`${this.config.baseURL}/embeddings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` }),
        },
        body: JSON.stringify(makeBody(false)),
      });
    }

    if (!response.ok) {
      throw new Error(`Embedding API error: ${response.status} ${response.statusText}`);
    }

    const result: EmbeddingResponse = await response.json();
    
    if (!result.data || result.data.length === 0) {
      throw new Error("No embedding data returned from API");
    }

    return result.data[0].embedding;
  }

  /**
   * Generate embeddings for multiple texts
   */
  async embedBatch(texts: string[]): Promise<number[][]> {
    const makeBody = (includeDimensions: boolean) => ({
      input: texts,
      model: this.config.model,
      ...(includeDimensions && this.config.dimensions ? { dimensions: this.config.dimensions } : {}),
    });

    let response = await fetch(`${this.config.baseURL}/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` }),
      },
      body: JSON.stringify(makeBody(true)),
    });

    if (!response.ok && response.status === 400 && this.config.dimensions) {
      response = await fetch(`${this.config.baseURL}/embeddings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` }),
        },
        body: JSON.stringify(makeBody(false)),
      });
    }

    if (!response.ok) {
      throw new Error(`Embedding API error: ${response.status} ${response.statusText}`);
    }

    const result: EmbeddingResponse = await response.json();
    
    if (!result.data) {
      throw new Error("No embedding data returned from API");
    }

    return result.data.map((item) => item.embedding);
  }
}

// Export default instance with config for convenience
export function createEmbedder(config: EmbeddingConfig): Embedder {
  return Embedder.create(config);
}
