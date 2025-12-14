import { useState, useEffect } from "react";
import {
  RefreshCw,
  FolderOpen,
  AlertCircle,
  CheckCircle2,
  Eye,
  EyeOff,
  Copy,
  FileText,
  Key,
} from "lucide-react";
import {
  useOAuthCredentials,
  useAllOAuthCredentials,
} from "@/hooks/useOAuthCredentials";
import { OAuthProvider, EnvVariable } from "@/lib/api/credentials";

type TabId = "kiro" | "gemini" | "qwen";

const tabs: { id: TabId; name: string; description: string }[] = [
  {
    id: "kiro",
    name: "Kiro Claude",
    description: "通过 Kiro OAuth 访问 Claude Sonnet 4.5",
  },
  {
    id: "gemini",
    name: "Gemini CLI",
    description: "通过 Gemini CLI OAuth 访问 Gemini 模型",
  },
  {
    id: "qwen",
    name: "通义千问",
    description: "通过 Qwen OAuth 访问通义千问",
  },
];

function CredentialPanel({ provider }: { provider: OAuthProvider }) {
  const {
    credentials,
    envVariables,
    loading,
    refreshing,
    error,
    reloadFromFile,
    refreshToken,
  } = useOAuthCredentials(provider);

  const [showEnv, setShowEnv] = useState(false);
  const [showValues, setShowValues] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  const handleReload = async () => {
    setMessage(null);
    try {
      await reloadFromFile();
      setMessage({ type: "success", text: "凭证加载成功！" });
    } catch (e) {
      setMessage({
        type: "error",
        text: `加载失败: ${e instanceof Error ? e.message : String(e)}`,
      });
    }
  };

  const handleRefresh = async () => {
    setMessage(null);
    try {
      await refreshToken();
      setMessage({ type: "success", text: "Token 刷新成功！" });
    } catch (e) {
      setMessage({
        type: "error",
        text: `刷新失败: ${e instanceof Error ? e.message : String(e)}`,
      });
    }
  };

  const copyValue = (key: string, value: string) => {
    navigator.clipboard.writeText(value);
    setCopied(key);
    setTimeout(() => setCopied(null), 2000);
  };

  const copyAllEnv = (vars: EnvVariable[]) => {
    navigator.clipboard.writeText(
      vars.map((v) => `${v.key}=${v.value}`).join("\n"),
    );
    setCopied("all");
    setTimeout(() => setCopied(null), 2000);
  };

  const isLoading = loading || refreshing;

  // Provider-specific extra info
  const renderExtraInfo = () => {
    if (!credentials?.extra) return null;
    const extra = credentials.extra as Record<string, unknown>;

    if (provider === "kiro") {
      return (
        <>
          <div>
            <span className="text-muted-foreground">区域:</span>
            <span className="ml-2">{(extra.region as string) || "未设置"}</span>
          </div>
          <div>
            <span className="text-muted-foreground">认证方式:</span>
            <span className="ml-2">
              {(extra.auth_method as string) || "social"}
            </span>
          </div>
        </>
      );
    }

    if (provider === "qwen" && extra.resource_url) {
      return (
        <div className="col-span-2">
          <span className="text-muted-foreground">Resource URL:</span>
          <code className="ml-2 rounded bg-muted px-2 py-0.5 text-xs break-all">
            {extra.resource_url as string}
          </code>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="space-y-4">
      {message && (
        <div
          className={`flex items-center gap-2 rounded-lg border p-3 text-sm ${
            message.type === "success"
              ? "border-green-500 bg-green-50 text-green-700 dark:bg-green-950/30"
              : "border-red-500 bg-red-50 text-red-700 dark:bg-red-950/30"
          }`}
        >
          {message.type === "success" ? (
            <CheckCircle2 className="h-4 w-4" />
          ) : (
            <AlertCircle className="h-4 w-4" />
          )}
          {message.text}
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 rounded-lg border border-red-500 bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950/30">
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}

      <div className="rounded-lg border bg-card p-4">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="font-semibold flex items-center gap-2">
            <Key className="h-4 w-4" />
            凭证状态
          </h3>
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <span
                className={`h-2 w-2 rounded-full ${
                  credentials?.is_valid
                    ? "bg-green-500"
                    : credentials?.loaded
                      ? "bg-yellow-500"
                      : "bg-gray-400"
                } animate-pulse`}
              />
              {credentials?.is_valid
                ? "有效"
                : credentials?.loaded
                  ? "已加载"
                  : "未加载"}
            </span>
          </div>
        </div>

        <div className="mb-4 grid grid-cols-2 gap-4 text-sm">
          <div className="col-span-2">
            <span className="text-muted-foreground">凭证路径:</span>
            <code className="ml-2 rounded bg-muted px-2 py-0.5 text-xs break-all">
              {credentials?.creds_path || "未知"}
            </code>
          </div>
          <div>
            <span className="text-muted-foreground">Access Token:</span>
            <span
              className={`ml-2 ${credentials?.has_access_token ? "text-green-600" : "text-red-500"}`}
            >
              {credentials?.has_access_token ? "已加载" : "未加载"}
            </span>
          </div>
          <div>
            <span className="text-muted-foreground">Refresh Token:</span>
            <span
              className={`ml-2 ${credentials?.has_refresh_token ? "text-green-600" : "text-red-500"}`}
            >
              {credentials?.has_refresh_token ? "已加载" : "未加载"}
            </span>
          </div>
          {renderExtraInfo()}
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            onClick={handleReload}
            disabled={isLoading}
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            <FolderOpen className="h-4 w-4" />
            {loading ? "加载中..." : "读取凭证"}
          </button>
          <button
            onClick={handleRefresh}
            disabled={isLoading || !credentials?.has_refresh_token}
            className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
            刷新 Token
          </button>
          <button
            onClick={() => setShowEnv(!showEnv)}
            className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
          >
            <FileText className="h-4 w-4" />
            {showEnv ? "隐藏" : "查看"} .env 变量
          </button>
        </div>
      </div>

      {/* Environment Variables */}
      {showEnv && (
        <div className="rounded-lg border bg-card p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="font-semibold">.env 环境变量</h3>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowValues(!showValues)}
                className="flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-muted"
              >
                {showValues ? (
                  <EyeOff className="h-3 w-3" />
                ) : (
                  <Eye className="h-3 w-3" />
                )}
                {showValues ? "隐藏值" : "显示值"}
              </button>
              <button
                onClick={() => copyAllEnv(envVariables)}
                className="flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-muted"
              >
                {copied === "all" ? (
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                ) : (
                  <Copy className="h-3 w-3" />
                )}
                复制全部
              </button>
            </div>
          </div>
          {envVariables.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              暂无环境变量，请先加载凭证
            </p>
          ) : (
            <div className="space-y-2 font-mono text-sm">
              {envVariables.map((v) => (
                <div
                  key={v.key}
                  className="flex items-center gap-2 rounded bg-muted p-2"
                >
                  <span className="text-blue-600 dark:text-blue-400 shrink-0">
                    {v.key}
                  </span>
                  <span>=</span>
                  <span className="flex-1 truncate text-muted-foreground">
                    {showValues ? v.value : v.masked}
                  </span>
                  <button
                    onClick={() => copyValue(v.key, v.value)}
                    className="rounded p-1 hover:bg-background shrink-0"
                  >
                    {copied === v.key ? (
                      <CheckCircle2 className="h-3 w-3 text-green-500" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function CredentialOverview() {
  const { credentials, reload } = useAllOAuthCredentials();

  useEffect(() => {
    reload();
  }, [reload]);

  return (
    <div className="grid grid-cols-3 gap-4">
      {credentials.map((cred) => {
        const tab = tabs.find((t) => t.id === cred.provider);
        return (
          <div
            key={cred.provider}
            className="rounded-lg border bg-card p-4 space-y-2"
          >
            <div className="flex items-center justify-between">
              <h4 className="font-medium">{tab?.name || cred.provider}</h4>
              <span
                className={`h-2 w-2 rounded-full ${
                  cred.is_valid
                    ? "bg-green-500"
                    : cred.loaded
                      ? "bg-yellow-500"
                      : "bg-gray-400"
                }`}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              {tab?.description || ""}
            </p>
            <div className="text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Access Token</span>
                <span className={cred.has_access_token ? "text-green-600" : "text-red-500"}>
                  {cred.has_access_token ? "已加载" : "未加载"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Refresh Token</span>
                <span className={cred.has_refresh_token ? "text-green-600" : "text-red-500"}>
                  {cred.has_refresh_token ? "已加载" : "未加载"}
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function CredentialsPage() {
  const [activeTab, setActiveTab] = useState<TabId>("kiro");

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">OAuth 凭证管理</h2>
        <p className="text-muted-foreground">
          管理 Kiro/Gemini/Qwen 的 OAuth 凭证
        </p>
      </div>

      {/* Overview Cards */}
      <CredentialOverview />

      {/* Tabs */}
      <div className="flex gap-2 border-b overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px whitespace-nowrap ${
              activeTab === tab.id
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.name}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <CredentialPanel provider={activeTab} />

      <p className="text-xs text-muted-foreground">
        系统每 5 秒自动检查凭证文件变化，如有更新会自动重新加载并记录日志
      </p>
    </div>
  );
}
