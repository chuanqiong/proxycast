import { invoke } from "@tauri-apps/api/core";

export interface Prompt {
  id: string;
  app_type: string;
  name: string;
  content: string;
  description?: string;
  enabled: boolean;
  createdAt?: number;
  updatedAt?: number;
}

export type AppType = "claude" | "codex" | "gemini";

export const promptsApi = {
  /** Get all prompts as a map (id -> Prompt) */
  getPrompts: (app: AppType): Promise<Record<string, Prompt>> =>
    invoke("get_prompts", { app }),

  /** Upsert a prompt (insert or update) */
  upsertPrompt: (app: AppType, id: string, prompt: Prompt): Promise<void> =>
    invoke("upsert_prompt", { app, id, prompt }),

  /** Add a new prompt */
  addPrompt: (prompt: Prompt): Promise<void> =>
    invoke("add_prompt", { prompt }),

  /** Update an existing prompt */
  updatePrompt: (prompt: Prompt): Promise<void> =>
    invoke("update_prompt", { prompt }),

  /** Delete a prompt */
  deletePrompt: (app: AppType, id: string): Promise<void> =>
    invoke("delete_prompt", { app, id }),

  /** Enable a prompt and sync to live file */
  enablePrompt: (app: AppType, id: string): Promise<void> =>
    invoke("enable_prompt", { app, id }),

  /** Import prompt from live file */
  importFromFile: (app: AppType): Promise<string> =>
    invoke("import_prompt_from_file", { app }),

  /** Get current live prompt file content */
  getCurrentFileContent: (app: AppType): Promise<string | null> =>
    invoke("get_current_prompt_file_content", { app }),

  /** Auto-import from live file if no prompts exist */
  autoImport: (app: AppType): Promise<number> =>
    invoke("auto_import_prompt", { app }),

  // Legacy API for compatibility
  switchPrompt: (appType: AppType, id: string): Promise<void> =>
    invoke("switch_prompt", { appType, id }),
};
