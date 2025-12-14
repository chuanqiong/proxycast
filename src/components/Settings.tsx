import { useState, useEffect } from "react";
import { Eye, EyeOff, Copy, Check } from "lucide-react";
import { getConfig, saveConfig, Config } from "@/hooks/useTauri";

export function Settings() {
  const [config, setConfig] = useState<Config | null>(null);
  const [showApiKey, setShowApiKey] = useState(false);
  const [copied, setCopied] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      const c = await getConfig();
      setConfig(c);
    } catch (e) {
      console.error(e);
    }
  };

  const handleSave = async () => {
    if (!config) return;
    setSaving(true);
    setMessage(null);
    try {
      await saveConfig(config);
      setMessage("设置已保存");
      setTimeout(() => setMessage(null), 2000);
    } catch (e: any) {
      setMessage(`保存失败: ${e.toString()}`);
    }
    setSaving(false);
  };

  const copyApiKey = () => {
    if (config) {
      navigator.clipboard.writeText(config.server.api_key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!config) {
    return <div>加载中...</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">设置</h2>
        <p className="text-muted-foreground">配置服务参数</p>
      </div>

      {message && (
        <div
          className={`rounded-lg border p-3 text-sm ${message.includes("失败") ? "border-red-500 bg-red-50 text-red-700" : "border-green-500 bg-green-50 text-green-700"}`}
        >
          {message}
        </div>
      )}

      <div className="max-w-md space-y-4 rounded-lg border bg-card p-6">
        <div>
          <label className="mb-1 block text-sm font-medium">监听地址</label>
          <input
            type="text"
            value={config.server.host}
            onChange={(e) =>
              setConfig({
                ...config,
                server: { ...config.server, host: e.target.value },
              })
            }
            className="w-full rounded-lg border bg-background px-3 py-2 text-sm"
          />
        </div>

        <div>
          <label className="mb-1 block text-sm font-medium">端口</label>
          <input
            type="number"
            value={config.server.port}
            onChange={(e) =>
              setConfig({
                ...config,
                server: {
                  ...config.server,
                  port: parseInt(e.target.value) || 3001,
                },
              })
            }
            className="w-full rounded-lg border bg-background px-3 py-2 text-sm"
          />
        </div>

        <div>
          <label className="mb-1 block text-sm font-medium">API Key</label>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <input
                type={showApiKey ? "text" : "password"}
                value={config.server.api_key}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    server: { ...config.server, api_key: e.target.value },
                  })
                }
                className="w-full rounded-lg border bg-background px-3 py-2 pr-20 text-sm"
              />
              <div className="absolute right-2 top-1/2 flex -translate-y-1/2 gap-1">
                <button
                  type="button"
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="rounded p-1 hover:bg-muted"
                  title={showApiKey ? "隐藏" : "显示"}
                >
                  {showApiKey ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
                <button
                  type="button"
                  onClick={copyApiKey}
                  className="rounded p-1 hover:bg-muted"
                  title="复制"
                >
                  {copied ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={handleSave}
          disabled={saving}
          className="w-full rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {saving ? "保存中..." : "保存设置"}
        </button>
      </div>
    </div>
  );
}
