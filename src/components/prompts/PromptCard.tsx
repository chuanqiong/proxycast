import { Check, Edit2, Trash2, FileText } from "lucide-react";
import { Prompt } from "@/lib/api/prompts";
import { cn } from "@/lib/utils";

interface PromptCardProps {
  prompt: Prompt;
  onToggle: (enabled: boolean) => void;
  onEdit: () => void;
  onDelete: () => void;
}

/** Toggle Switch Component */
function ToggleSwitch({
  enabled,
  onChange,
  disabled = false,
}: {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      disabled={disabled}
      onClick={(e) => {
        e.stopPropagation();
        onChange(!enabled);
      }}
      className={cn(
        "relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary/20",
        enabled ? "bg-primary" : "bg-muted-foreground/30",
        disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"
      )}
    >
      <span
        className={cn(
          "inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform",
          enabled ? "translate-x-5" : "translate-x-1"
        )}
      />
    </button>
  );
}

export function PromptCard({
  prompt,
  onToggle,
  onEdit,
  onDelete,
}: PromptCardProps) {
  return (
    <div
      className={cn(
        "relative rounded-lg border p-4 transition-all",
        prompt.enabled
          ? "border-primary bg-primary/5 ring-1 ring-primary"
          : "hover:border-muted-foreground/50"
      )}
    >
      {prompt.enabled && (
        <div className="absolute -top-2 -right-2 rounded-full bg-primary p-1">
          <Check className="h-3 w-3 text-primary-foreground" />
        </div>
      )}

      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <ToggleSwitch
            enabled={prompt.enabled}
            onChange={onToggle}
          />
          <div className="h-8 w-8 rounded-lg bg-muted flex items-center justify-center">
            <FileText className="h-4 w-4" />
          </div>
          <div>
            <h3 className="font-medium">{prompt.name}</h3>
            {prompt.description && (
              <span className="text-xs text-muted-foreground">
                {prompt.description}
              </span>
            )}
          </div>
        </div>

        <div className="flex gap-1">
          <button
            onClick={onEdit}
            className="p-1.5 rounded hover:bg-muted"
            title="编辑"
          >
            <Edit2 className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={onDelete}
            disabled={prompt.enabled}
            className={cn(
              "p-1.5 rounded",
              prompt.enabled
                ? "opacity-30 cursor-not-allowed"
                : "hover:bg-destructive/10 text-destructive"
            )}
            title={prompt.enabled ? "无法删除已启用的提示词" : "删除"}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      <div className="mt-3 p-2 rounded bg-muted/50 max-h-24 overflow-auto">
        <pre className="text-xs text-muted-foreground whitespace-pre-wrap font-mono">
          {prompt.content.slice(0, 200)}
          {prompt.content.length > 200 && "..."}
        </pre>
      </div>

      <div className="mt-3">
        {prompt.enabled ? (
          <span className="text-sm text-primary font-medium">已启用 (同步到配置文件)</span>
        ) : (
          <span className="text-sm text-muted-foreground">未启用</span>
        )}
      </div>
    </div>
  );
}
