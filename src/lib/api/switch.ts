import { invoke } from "@tauri-apps/api/core";

export interface Provider {
  id: string;
  app_type: string;
  name: string;
  settings_config: Record<string, unknown>;
  category?: string;
  icon?: string;
  icon_color?: string;
  notes?: string;
  created_at?: number;
  sort_index?: number;
  is_current: boolean;
}

// proxycast 保留用于内部配置存储，但不在 UI 的 Tab 中显示
export type AppType = "claude" | "codex" | "gemini" | "proxycast";

export const switchApi = {
  getProviders: (appType: AppType): Promise<Provider[]> =>
    invoke("get_switch_providers", { appType }),

  getCurrentProvider: (appType: AppType): Promise<Provider | null> =>
    invoke("get_current_switch_provider", { appType }),

  addProvider: (provider: Provider): Promise<void> =>
    invoke("add_switch_provider", { provider }),

  updateProvider: (provider: Provider): Promise<void> =>
    invoke("update_switch_provider", { provider }),

  deleteProvider: (appType: AppType, id: string): Promise<void> =>
    invoke("delete_switch_provider", { appType, id }),

  switchProvider: (appType: AppType, id: string): Promise<void> =>
    invoke("switch_provider", { appType, id }),

  /** 读取当前生效的配置（从实际配置文件读取） */
  readLiveSettings: (appType: AppType): Promise<Record<string, unknown>> =>
    invoke("read_live_provider_settings", { appType }),
};
