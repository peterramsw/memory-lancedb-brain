declare module "openclaw/plugin-sdk" {
  export interface OpenClawPluginApi {
    config?: unknown;
    log: {
      info(message: string): void;
      error(message: string): void;
      warn?(message: string): void;
      debug?(message: string): void;
    };
  }
}
