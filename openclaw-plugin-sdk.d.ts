declare module "openclaw/plugin-sdk" {
  export interface OpenClawPluginToolContext {
    config?: unknown;
    workspaceDir?: string;
    agentDir?: string;
    agentId?: string;
    sessionKey?: string;
    sessionId?: string;
    messageChannel?: string;
    agentAccountId?: string;
    requesterSenderId?: string;
    senderIsOwner?: boolean;
    sandboxed?: boolean;
  }

  export interface OpenClawPluginApi {
    id?: string;
    name?: string;
    config?: unknown;
    pluginConfig?: unknown;
    logger?: {
      info?(message: string): void;
      error?(message: string): void;
      warn?(message: string): void;
      debug?(message: string): void;
    };
    resolvePath?(input: string): string;
    registerTool?(tool: unknown, opts?: unknown): void;
    registerCommand?(command: unknown): void;
    registerContextEngine?(id: string, factory: unknown): void;
    on?(hookName: string, handler: (...args: unknown[]) => unknown, opts?: unknown): void;
  }
}
