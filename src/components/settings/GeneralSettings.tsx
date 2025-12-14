import { useState, useEffect } from "react";
import { Moon, Sun, Monitor } from "lucide-react";
import { cn } from "@/lib/utils";

type Theme = "light" | "dark" | "system";

export function GeneralSettings() {
  const [theme, setTheme] = useState<Theme>("system");
  const [launchOnStartup, setLaunchOnStartup] = useState(false);
  const [minimizeToTray, setMinimizeToTray] = useState(true);

  useEffect(() => {
    // 读取当前主题
    const savedTheme = localStorage.getItem("theme") as Theme | null;
    if (savedTheme) {
      setTheme(savedTheme);
    }
  }, []);

  const handleThemeChange = (newTheme: Theme) => {
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);

    // 应用主题
    const root = document.documentElement;
    if (newTheme === "system") {
      const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.classList.toggle("dark", systemDark);
    } else {
      root.classList.toggle("dark", newTheme === "dark");
    }
  };

  const themeOptions = [
    { id: "light" as Theme, label: "浅色", icon: Sun },
    { id: "dark" as Theme, label: "深色", icon: Moon },
    { id: "system" as Theme, label: "跟随系统", icon: Monitor },
  ];

  return (
    <div className="space-y-6 max-w-2xl">
      {/* 主题设置 */}
      <div className="space-y-3">
        <div>
          <h3 className="text-sm font-medium">主题</h3>
          <p className="text-xs text-muted-foreground">选择界面显示主题</p>
        </div>
        <div className="flex gap-2">
          {themeOptions.map((option) => (
            <button
              key={option.id}
              onClick={() => handleThemeChange(option.id)}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors",
                theme === option.id
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border hover:border-muted-foreground/50"
              )}
            >
              <option.icon className="h-4 w-4" />
              <span className="text-sm">{option.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* 启动设置 */}
      <div className="space-y-4">
        <div>
          <h3 className="text-sm font-medium">启动行为</h3>
          <p className="text-xs text-muted-foreground">配置应用启动和关闭行为</p>
        </div>

        <div className="space-y-3">
          <label className="flex items-center justify-between p-3 rounded-lg border cursor-pointer hover:bg-muted/50">
            <div>
              <span className="text-sm font-medium">开机自启动</span>
              <p className="text-xs text-muted-foreground">系统启动时自动运行 ProxyCast</p>
            </div>
            <input
              type="checkbox"
              checked={launchOnStartup}
              onChange={(e) => setLaunchOnStartup(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300"
            />
          </label>

          <label className="flex items-center justify-between p-3 rounded-lg border cursor-pointer hover:bg-muted/50">
            <div>
              <span className="text-sm font-medium">关闭时最小化到托盘</span>
              <p className="text-xs text-muted-foreground">点击关闭按钮时最小化而不是退出</p>
            </div>
            <input
              type="checkbox"
              checked={minimizeToTray}
              onChange={(e) => setMinimizeToTray(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300"
            />
          </label>
        </div>
      </div>
    </div>
  );
}
