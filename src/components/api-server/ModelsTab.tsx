import { useState, useEffect } from "react";
import { Cpu, RefreshCw, Copy, Check, Search } from "lucide-react";
import { getAvailableModels, ModelInfo } from "@/hooks/useTauri";

// 模型分组配置
const MODEL_GROUPS: Record<
  string,
  { name: string; color: string; models: string[] }
> = {
  kiro: {
    name: "Kiro Claude",
    color: "bg-purple-100 text-purple-700",
    models: [
      "claude-sonnet-4-5",
      "claude-sonnet-4-5-20250514",
      "claude-sonnet-4-5-20250929",
      "claude-3-7-sonnet-20250219",
      "claude-3-5-sonnet-latest",
      "claude-3-5-sonnet-20241022",
      "claude-opus-4-5-20250514",
      "claude-haiku-4-5-20250514",
    ],
  },
  gemini: {
    name: "Gemini CLI",
    color: "bg-blue-100 text-blue-700",
    models: [
      "gemini-2.5-flash",
      "gemini-2.5-flash-lite",
      "gemini-2.5-pro",
      "gemini-2.5-pro-preview-06-05",
      "gemini-3-pro-preview",
      "gemini-2.0-flash-exp",
    ],
  },
  qwen: {
    name: "通义千问",
    color: "bg-orange-100 text-orange-700",
    models: [
      "qwen3-coder-plus",
      "qwen3-coder-flash",
      "qwen-coder-plus",
      "qwen-coder-turbo",
    ],
  },
  openai: {
    name: "OpenAI",
    color: "bg-green-100 text-green-700",
    models: [
      "gpt-4o",
      "gpt-4o-mini",
      "gpt-4-turbo",
      "gpt-4",
      "gpt-3.5-turbo",
      "o1-preview",
      "o1-mini",
    ],
  },
};

export function ModelsTab() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [selectedGroup, setSelectedGroup] = useState<string | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await getAvailableModels();
      setModels(data || []);
    } catch (e: any) {
      setError(e.toString());
      setModels([]);
    }

    setLoading(false);
  };

  const copyModelId = (id: string) => {
    navigator.clipboard.writeText(id);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const getModelGroup = (modelId: string): string | null => {
    for (const [groupId, group] of Object.entries(MODEL_GROUPS)) {
      if (
        group.models.some((m) =>
          modelId.toLowerCase().includes(m.toLowerCase().split("-")[0]),
        )
      ) {
        return groupId;
      }
    }
    // 根据 owned_by 判断
    const model = models.find((m) => m.id === modelId);
    if (model?.owned_by === "anthropic") return "kiro";
    if (model?.owned_by === "google") return "gemini";
    if (model?.owned_by === "alibaba") return "qwen";
    if (model?.owned_by === "openai") return "openai";
    return null;
  };

  const getGroupBadge = (groupId: string | null) => {
    if (!groupId || !MODEL_GROUPS[groupId]) return null;
    const group = MODEL_GROUPS[groupId];
    return (
      <span
        className={`rounded px-2 py-0.5 text-xs font-medium ${group.color}`}
      >
        {group.name}
      </span>
    );
  };

  // 过滤模型
  const filteredModels = models.filter((model) => {
    const matchesSearch = model.id.toLowerCase().includes(search.toLowerCase());
    const matchesGroup =
      !selectedGroup || getModelGroup(model.id) === selectedGroup;
    return matchesSearch && matchesGroup;
  });

  // 按 provider 分组统计
  const groupCounts = models.reduce(
    (acc, model) => {
      const group = getModelGroup(model.id);
      if (group) {
        acc[group] = (acc[group] || 0) + 1;
      }
      return acc;
    },
    {} as Record<string, number>,
  );

  return (
    <div className="space-y-6">
      {error && (
        <div className="rounded-lg border border-red-500 bg-red-50 p-4 text-red-700">
          {error}
        </div>
      )}

      {/* 搜索和过滤 */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="搜索模型..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-lg border bg-background pl-10 pr-4 py-2 text-sm"
          />
        </div>
        <button
          onClick={fetchModels}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          刷新
        </button>
      </div>

      {/* Provider 过滤标签 */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setSelectedGroup(null)}
          className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
            !selectedGroup
              ? "bg-primary text-primary-foreground"
              : "bg-muted hover:bg-muted/80"
          }`}
        >
          全部 ({models.length})
        </button>
        {Object.entries(MODEL_GROUPS).map(([groupId, group]) => {
          const count = groupCounts[groupId] || 0;
          if (count === 0) return null;
          return (
            <button
              key={groupId}
              onClick={() =>
                setSelectedGroup(selectedGroup === groupId ? null : groupId)
              }
              className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                selectedGroup === groupId
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted hover:bg-muted/80"
              }`}
            >
              {group.name} ({count})
            </button>
          );
        })}
      </div>

      {/* 模型列表 */}
      <div className="rounded-lg border bg-card">
        <div className="border-b px-4 py-3">
          <div className="flex items-center justify-between">
            <span className="font-medium">模型列表</span>
            <span className="text-sm text-muted-foreground">
              {filteredModels.length} 个模型
            </span>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : filteredModels.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <Cpu className="h-12 w-12 mb-2 opacity-50" />
            <p>暂无模型数据</p>
          </div>
        ) : (
          <div className="divide-y">
            {filteredModels.map((model) => (
              <div
                key={model.id}
                className="flex items-center justify-between px-4 py-3 hover:bg-muted/50"
              >
                <div className="flex items-center gap-3">
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <div className="flex items-center gap-2">
                      <code className="font-medium">{model.id}</code>
                      {getGroupBadge(getModelGroup(model.id))}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {model.owned_by}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => copyModelId(model.id)}
                  className="rounded p-2 hover:bg-muted"
                  title="复制模型 ID"
                >
                  {copied === model.id ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 使用说明 */}
      <div className="rounded-lg border bg-card p-4">
        <h3 className="mb-2 font-semibold">使用说明</h3>
        <div className="space-y-2 text-sm text-muted-foreground">
          <p>• 点击模型 ID 右侧的复制按钮可快速复制模型名称</p>
          <p>
            • 在 API 请求中使用{" "}
            <code className="rounded bg-muted px-1">model</code> 参数指定模型
          </p>
          <p>• 不同 Provider 支持的模型不同，请确保已配置对应的凭证</p>
        </div>
      </div>
    </div>
  );
}
