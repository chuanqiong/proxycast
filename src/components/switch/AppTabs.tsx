import { cn } from "@/lib/utils";
import { AppType } from "@/lib/api/switch";

interface AppTabsProps {
  activeApp: AppType;
  onAppChange: (app: AppType) => void;
}

const apps: { id: AppType; label: string; description: string }[] = [
  { id: "claude", label: "Claude Code", description: "Claude CLI 配置" },
  { id: "codex", label: "Codex", description: "OpenAI Codex CLI" },
  { id: "gemini", label: "Gemini", description: "Google Gemini CLI" },
];

export function AppTabs({ activeApp, onAppChange }: AppTabsProps) {
  return (
    <div className="flex gap-2 border-b pb-2">
      {apps.map((app) => (
        <button
          key={app.id}
          onClick={() => onAppChange(app.id)}
          className={cn(
            "px-4 py-2 rounded-t-lg text-sm font-medium transition-colors",
            activeApp === app.id
              ? "bg-primary text-primary-foreground"
              : "hover:bg-muted text-muted-foreground"
          )}
          title={app.description}
        >
          {app.label}
        </button>
      ))}
    </div>
  );
}
