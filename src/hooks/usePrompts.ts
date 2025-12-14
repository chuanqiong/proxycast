import { useState, useCallback, useRef } from "react";
import { promptsApi, Prompt, AppType } from "@/lib/api/prompts";

export function usePrompts(appType: AppType) {
  const [prompts, setPrompts] = useState<Record<string, Prompt>>({});
  const [currentFileContent, setCurrentFileContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const hasAutoImported = useRef<Record<string, boolean>>({});

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      // Auto-import on first load if no prompts exist
      if (!hasAutoImported.current[appType]) {
        hasAutoImported.current[appType] = true;
        try {
          const imported = await promptsApi.autoImport(appType);
          if (imported > 0) {
            console.log(`Auto-imported ${imported} prompt(s) for ${appType}`);
          }
        } catch (e) {
          console.log("Auto-import skipped:", e);
        }
      }

      const data = await promptsApi.getPrompts(appType);
      setPrompts(data);

      // Also load current file content
      try {
        const content = await promptsApi.getCurrentFileContent(appType);
        setCurrentFileContent(content);
      } catch {
        setCurrentFileContent(null);
      }
    } catch (error) {
      console.error("Failed to load prompts:", error);
    } finally {
      setLoading(false);
    }
  }, [appType]);

  const savePrompt = useCallback(
    async (id: string, prompt: Prompt) => {
      await promptsApi.upsertPrompt(appType, id, prompt);
      await reload();
    },
    [appType, reload],
  );

  const deletePrompt = useCallback(
    async (id: string) => {
      await promptsApi.deletePrompt(appType, id);
      await reload();
    },
    [appType, reload],
  );

  const enablePrompt = useCallback(
    async (id: string) => {
      await promptsApi.enablePrompt(appType, id);
      await reload();
    },
    [appType, reload],
  );

  const toggleEnabled = useCallback(
    async (id: string, enabled: boolean) => {
      // Optimistic update
      const previousPrompts = prompts;

      if (enabled) {
        // If enabling, first disable all others
        const updatedPrompts = Object.keys(prompts).reduce(
          (acc, key) => {
            acc[key] = {
              ...prompts[key],
              enabled: key === id,
            };
            return acc;
          },
          {} as Record<string, Prompt>,
        );
        setPrompts(updatedPrompts);
      } else {
        setPrompts((prev) => ({
          ...prev,
          [id]: {
            ...prev[id],
            enabled: false,
          },
        }));
      }

      try {
        if (enabled) {
          await promptsApi.enablePrompt(appType, id);
        } else {
          // Disable by updating with enabled=false
          await promptsApi.upsertPrompt(appType, id, {
            ...prompts[id],
            enabled: false,
          });
        }
        await reload();
      } catch (error) {
        // Rollback on failure
        setPrompts(previousPrompts);
        throw error;
      }
    },
    [appType, prompts, reload],
  );

  const importFromFile = useCallback(async () => {
    const id = await promptsApi.importFromFile(appType);
    await reload();
    return id;
  }, [appType, reload]);

  return {
    prompts,
    loading,
    currentFileContent,
    reload,
    savePrompt,
    deletePrompt,
    enablePrompt,
    toggleEnabled,
    importFromFile,
  };
}
