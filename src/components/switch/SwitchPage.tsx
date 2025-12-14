import { useState } from "react";
import { AppType } from "@/lib/api/switch";
import { AppTabs } from "./AppTabs";
import { ProviderList } from "./ProviderList";

export function SwitchPage() {
  const [activeApp, setActiveApp] = useState<AppType>("claude");

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Switch</h2>
        <p className="text-muted-foreground">
          管理和切换不同应用的 Provider 配置
        </p>
      </div>

      <AppTabs activeApp={activeApp} onAppChange={setActiveApp} />
      <ProviderList appType={activeApp} />
    </div>
  );
}
