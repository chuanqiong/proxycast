import { useState, useEffect, useRef } from "react";
import { Trash2, Download } from "lucide-react";
import { getLogs, clearLogs, LogEntry } from "@/hooks/useTauri";

export function LogsTab() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, autoScroll]);

  const fetchLogs = async () => {
    try {
      const l = await getLogs();
      setLogs(l);
    } catch (e) {
      // 如果后端还没实现，使用空数组
      console.error(e);
    }
  };

  const handleClear = async () => {
    try {
      await clearLogs();
      setLogs([]);
    } catch {
      setLogs([]);
    }
  };

  const handleExport = () => {
    const content = logs
      .map(
        (l) =>
          `[${new Date(l.timestamp).toLocaleString()}] [${l.level.toUpperCase()}] ${l.message}`,
      )
      .join("\n");

    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `proxycast-logs-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case "error":
        return "text-red-500";
      case "warn":
        return "text-yellow-500";
      case "debug":
        return "text-gray-400";
      default:
        return "text-blue-500";
    }
  };

  const getLevelBg = (level: string) => {
    switch (level) {
      case "error":
        return "bg-red-500/10";
      case "warn":
        return "bg-yellow-500/10";
      default:
        return "";
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-end gap-2">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
            className="rounded"
          />
          自动滚动
        </label>
        <button
          onClick={handleExport}
          className="flex items-center gap-2 rounded-lg border px-3 py-2 text-sm hover:bg-muted"
        >
          <Download className="h-4 w-4" />
          导出
        </button>
        <button
          onClick={handleClear}
          className="flex items-center gap-2 rounded-lg border px-3 py-2 text-sm hover:bg-muted"
        >
          <Trash2 className="h-4 w-4" />
          清空
        </button>
      </div>

      <div className="rounded-lg border bg-card">
        <div className="max-h-[600px] overflow-auto p-4 font-mono text-sm">
          {logs.length === 0 ? (
            <p className="text-center text-muted-foreground">
              暂无日志，启动服务后将显示请求日志
            </p>
          ) : (
            logs.map((log, i) => (
              <div
                key={i}
                className={`flex gap-2 py-1 px-2 rounded ${getLevelBg(log.level)}`}
              >
                <span className="text-muted-foreground shrink-0">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                <span
                  className={`font-medium shrink-0 ${getLevelColor(log.level)}`}
                >
                  [{log.level.toUpperCase()}]
                </span>
                <span className="break-all">{log.message}</span>
              </div>
            ))
          )}
          <div ref={logsEndRef} />
        </div>
      </div>
    </div>
  );
}
