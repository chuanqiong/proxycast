import { useState, useEffect, useCallback } from "react";
import { switchApi, Provider, AppType } from "@/lib/api/switch";

export function useSwitch(appType: AppType) {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [currentProvider, setCurrentProvider] = useState<Provider | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchProviders = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [list, current] = await Promise.all([
        switchApi.getProviders(appType),
        switchApi.getCurrentProvider(appType),
      ]);
      setProviders(list);
      setCurrentProvider(current);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [appType]);

  useEffect(() => {
    fetchProviders();
  }, [fetchProviders]);

  const addProvider = async (provider: Omit<Provider, "id" | "is_current" | "created_at">) => {
    const newProvider: Provider = {
      ...provider,
      id: crypto.randomUUID(),
      app_type: appType,
      is_current: false,
      created_at: Date.now(),
    };
    await switchApi.addProvider(newProvider);
    await fetchProviders();
  };

  const updateProvider = async (provider: Provider) => {
    await switchApi.updateProvider(provider);
    await fetchProviders();
  };

  const deleteProvider = async (id: string) => {
    await switchApi.deleteProvider(appType, id);
    await fetchProviders();
  };

  const switchToProvider = async (id: string) => {
    await switchApi.switchProvider(appType, id);
    await fetchProviders();
  };

  return {
    providers,
    currentProvider,
    loading,
    error,
    addProvider,
    updateProvider,
    deleteProvider,
    switchToProvider,
    refresh: fetchProviders,
  };
}
