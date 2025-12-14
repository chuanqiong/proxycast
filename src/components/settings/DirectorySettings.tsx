import { useState } from "react";
import { Folder, RotateCcw } from "lucide-react";

interface DirectoryConfig {
  claudeConfigDir: string;
  codexConfigDir: string;
  geminiConfigDir: string;
}

const defaultDirs: DirectoryConfig = {
  claudeConfigDir: "~/.claude",
  codexConfigDir: "~/.codex",
  geminiConfigDir: "~/.gemini",
};

export function DirectorySettings() {
  const [dirs, setDirs] = useState<DirectoryConfig>(defaultDirs);
  const [saving, setSaving] = useState(false);

  const handleReset = (key: keyof DirectoryConfig) => {
    setDirs((prev) => ({ ...prev, [key]: defaultDirs[key] }));
  };

  const handleSave = async () => {
    setSaving(true);
    // TODO: 保存目录配置到后端
    await new Promise((resolve) => setTimeout(resolve, 500));
    setSaving(false);
  };

  const directoryItems = [
    {
      key: "claudeConfigDir" as const,
      label: "Claude 配置目录",
      description: "Claude Code 配置文件存储位置",
    },
    {
      key: "codexConfigDir" as const,
      label: "Codex 配置目录",
      description: "Codex CLI 配置文件存储位置",
    },
    {
      key: "geminiConfigDir" as const,
      label: "Gemini 配置目录",
      description: "Gemini CLI 配置文件存储位置",
    },
  ];

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h3 className="text-sm font-medium">配置目录</h3>
        <p className="text-xs text-muted-foreground">
          自定义各应用的配置文件目录位置。修改后需要重启应用生效。
        </p>
      </div>

      <div className="space-y-4">
        {directoryItems.map((item) => (
          <div key={item.key} className="p-4 rounded-lg border space-y-2">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium">{item.label}</label>
                <p className="text-xs text-muted-foreground">{item.description}</p>
              </div>
              <button
                onClick={() => handleReset(item.key)}
                className="p-1.5 rounded hover:bg-muted text-muted-foreground"
                title="重置为默认"
              >
                <RotateCcw className="h-4 w-4" />
              </button>
            </div>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Folder className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <input
                  type="text"
                  value={dirs[item.key]}
                  onChange={(e) =>
                    setDirs((prev) => ({ ...prev, [item.key]: e.target.value }))
                  }
                  className="w-full pl-9 pr-3 py-2 rounded-lg border bg-background text-sm font-mono focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none"
                  placeholder={defaultDirs[item.key]}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="flex justify-end">
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
        >
          {saving ? "保存中..." : "保存设置"}
        </button>
      </div>

      {/* 数据管理 */}
      <div className="pt-6 border-t space-y-4">
        <div>
          <h3 className="text-sm font-medium">数据管理</h3>
          <p className="text-xs text-muted-foreground">导入或导出配置数据</p>
        </div>

        <div className="flex gap-3">
          <button className="px-4 py-2 rounded-lg border text-sm hover:bg-muted">
            导出配置
          </button>
          <button className="px-4 py-2 rounded-lg border text-sm hover:bg-muted">
            导入配置
          </button>
        </div>
      </div>
    </div>
  );
}
