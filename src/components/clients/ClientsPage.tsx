import { useState } from "react";
import { AppType } from "@/lib/api/switch";
import { AppTabs } from "./AppTabs";
import { ProviderList } from "./ProviderList";

export function ClientsPage() {
  const [activeApp, setActiveApp] = useState<AppType>("claude");

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">AI Clients</h2>
        <p className="text-muted-foreground">
          管理 Claude Code / Codex / Gemini CLI 的 Provider 配置
        </p>
      </div>

      <AppTabs activeApp={activeApp} onAppChange={setActiveApp} />
      <ProviderList appType={activeApp} />
    </div>
  );
}
