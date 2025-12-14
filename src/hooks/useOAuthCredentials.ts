import { useState, useCallback, useEffect, useRef } from "react";
import {
  credentialsApi,
  OAuthProvider,
  OAuthCredentialStatus,
  EnvVariable,
} from "@/lib/api/credentials";

export function useOAuthCredentials(provider: OAuthProvider) {
  const [credentials, setCredentials] = useState<OAuthCredentialStatus | null>(
    null,
  );
  const [envVariables, setEnvVariables] = useState<EnvVariable[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastHashRef = useRef<string>("");

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [creds, vars] = await Promise.all([
        credentialsApi.getCredentials(provider),
        credentialsApi.getEnvVariables(provider),
      ]);
      setCredentials(creds);
      setEnvVariables(vars);

      // Update hash for change detection
      const hash = await credentialsApi.getTokenFileHash(provider);
      lastHashRef.current = hash;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [provider]);

  const reloadFromFile = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await credentialsApi.reloadCredentials(provider);
      await reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [provider, reload]);

  const refreshToken = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      await credentialsApi.refreshToken(provider);
      await reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRefreshing(false);
    }
  }, [provider, reload]);

  // Auto-check for file changes
  const checkForChanges = useCallback(async () => {
    if (!lastHashRef.current) return;

    try {
      const result = await credentialsApi.checkAndReload(
        provider,
        lastHashRef.current,
      );
      if (result.changed && result.reloaded) {
        lastHashRef.current = result.new_hash;
        await reload();
      }
    } catch (e) {
      console.error("Error checking for credential changes:", e);
    }
  }, [provider, reload]);

  // Initial load
  useEffect(() => {
    reload();
  }, [reload]);

  // Periodic check for file changes
  useEffect(() => {
    const interval = setInterval(checkForChanges, 5000);
    return () => clearInterval(interval);
  }, [checkForChanges]);

  return {
    credentials,
    envVariables,
    loading,
    refreshing,
    error,
    reload,
    reloadFromFile,
    refreshToken,
    checkForChanges,
  };
}

// Hook to get all credentials at once
export function useAllOAuthCredentials() {
  const [credentials, setCredentials] = useState<OAuthCredentialStatus[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const creds = await credentialsApi.getAllCredentials();
      setCredentials(creds);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  return {
    credentials,
    loading,
    error,
    reload,
  };
}
