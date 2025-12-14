import { invoke } from "@tauri-apps/api/core";

export type OAuthProvider = "kiro" | "gemini" | "qwen";

export interface OAuthCredentialStatus {
  provider: string;
  loaded: boolean;
  has_access_token: boolean;
  has_refresh_token: boolean;
  is_valid: boolean;
  expiry_info: string | null;
  creds_path: string;
  extra: Record<string, unknown>;
}

export interface EnvVariable {
  key: string;
  value: string;
  masked: string;
}

export interface CheckResult {
  changed: boolean;
  new_hash: string;
  reloaded: boolean;
}

export const credentialsApi = {
  /** Get credentials status for a specific provider */
  getCredentials: (provider: OAuthProvider): Promise<OAuthCredentialStatus> =>
    invoke("get_oauth_credentials", { provider }),

  /** Get all OAuth credentials at once */
  getAllCredentials: (): Promise<OAuthCredentialStatus[]> =>
    invoke("get_all_oauth_credentials"),

  /** Reload credentials from file */
  reloadCredentials: (provider: OAuthProvider): Promise<string> =>
    invoke("reload_oauth_credentials", { provider }),

  /** Refresh OAuth token */
  refreshToken: (provider: OAuthProvider): Promise<string> =>
    invoke("refresh_oauth_token", { provider }),

  /** Get environment variables for a provider */
  getEnvVariables: (provider: OAuthProvider): Promise<EnvVariable[]> =>
    invoke("get_oauth_env_variables", { provider }),

  /** Get token file hash for change detection */
  getTokenFileHash: (provider: OAuthProvider): Promise<string> =>
    invoke("get_oauth_token_file_hash", { provider }),

  /** Check and reload credentials if file changed */
  checkAndReload: (
    provider: OAuthProvider,
    lastHash: string,
  ): Promise<CheckResult> =>
    invoke("check_and_reload_oauth_credentials", {
      provider,
      lastHash,
    }),
};
