import React, { useState } from "react";
import { X } from "lucide-react";
import { Prompt, AppType } from "@/lib/api/prompts";

interface PromptFormProps {
  appType: AppType;
  prompt: Prompt | null;
  onSave: (data: Omit<Prompt, "id" | "createdAt">) => Promise<void>;
  onCancel: () => void;
}

export function PromptForm({ appType, prompt, onSave, onCancel }: PromptFormProps) {
  const [name, setName] = useState(prompt?.name || "");
  const [content, setContent] = useState(prompt?.content || "");
  const [description, setDescription] = useState(prompt?.description || "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    try {
      setSaving(true);
      await onSave({
        app_type: appType,
        name,
        content,
        description: description || undefined,
        enabled: prompt?.enabled || false,
        updatedAt: Math.floor(Date.now() / 1000),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-lg w-full max-w-2xl max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between p-4 border-b">
          <h3 className="font-semibold">
            {prompt ? "编辑 Prompt" : "添加 Prompt"}
          </h3>
          <button onClick={onCancel} className="p-1 hover:bg-muted rounded">
            <X className="h-5 w-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">名称</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border bg-background"
              placeholder="Prompt 名称"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">描述</label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border bg-background"
              placeholder="可选描述"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">内容</label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border bg-background font-mono text-sm"
              rows={15}
              placeholder="输入系统提示词内容..."
              required
            />
          </div>

          {error && <p className="text-sm text-destructive">{error}</p>}

          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-2 rounded-lg border hover:bg-muted"
            >
              取消
            </button>
            <button
              type="submit"
              disabled={saving || !name || !content}
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground disabled:opacity-50"
            >
              {saving ? "保存中..." : "保存"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
