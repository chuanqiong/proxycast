import { useState, useCallback } from "react";
import { Activity, Download, BarChart3, List, Trash2 } from "lucide-react";
import {
  FlowList,
  FlowFilters,
  FlowDetail,
  FlowStats,
  ExportDialog,
  CleanupDialog,
} from "@/components/flow-monitor";
import {
  type LLMFlow,
  type FlowFilter,
  type ExportFormat,
} from "@/lib/api/flowMonitor";
import { cn } from "@/lib/utils";

type ViewMode = "list" | "stats";

export function FlowMonitorPage() {
  // 视图模式
  const [viewMode, setViewMode] = useState<ViewMode>("list");

  // 过滤条件
  const [filter, setFilter] = useState<FlowFilter>({});

  // 选中的 Flow
  const [selectedFlow, setSelectedFlow] = useState<LLMFlow | null>(null);

  // 导出对话框
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportFlowIds, setExportFlowIds] = useState<string[]>([]);

  // 清理对话框
  const [cleanupDialogOpen, setCleanupDialogOpen] = useState(false);

  // 刷新计数器（用于触发子组件刷新）
  const [refreshKey, setRefreshKey] = useState(0);

  // 处理 Flow 选择
  const handleFlowSelect = useCallback((flow: LLMFlow) => {
    setSelectedFlow(flow);
  }, []);

  // 返回列表
  const handleBackToList = useCallback(() => {
    setSelectedFlow(null);
  }, []);

  // 导出单个 Flow
  const handleExportFlow = useCallback(
    (flowId: string, _format: ExportFormat) => {
      setExportFlowIds([flowId]);
      setExportDialogOpen(true);
    },
    [],
  );

  // 批量导出
  const handleBatchExport = useCallback(() => {
    setExportFlowIds([]);
    setExportDialogOpen(true);
  }, []);

  // 导出成功
  const handleExportSuccess = useCallback((filename: string) => {
    console.log("导出成功:", filename);
  }, []);

  // 清理成功
  const handleCleanupSuccess = useCallback(() => {
    // 清理成功后刷新数据
    setRefreshKey((prev) => prev + 1);
  }, []);

  return (
    <div className="space-y-6">
      {/* 页面头部 */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="shrink-0">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Activity className="h-6 w-6" />
            Flow Monitor
          </h2>
          <p className="text-muted-foreground">
            监控和分析 LLM API 请求/响应流量
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {/* 视图切换 */}
          <div className="flex items-center rounded-lg border p-1">
            <button
              onClick={() => setViewMode("list")}
              className={cn(
                "flex items-center gap-1 px-3 py-1.5 rounded text-sm transition-colors",
                viewMode === "list"
                  ? "bg-primary text-primary-foreground"
                  : "hover:bg-muted",
              )}
            >
              <List className="h-4 w-4" />
              列表
            </button>
            <button
              onClick={() => setViewMode("stats")}
              className={cn(
                "flex items-center gap-1 px-3 py-1.5 rounded text-sm transition-colors",
                viewMode === "stats"
                  ? "bg-primary text-primary-foreground"
                  : "hover:bg-muted",
              )}
            >
              <BarChart3 className="h-4 w-4" />
              统计
            </button>
          </div>

          {/* 导出按钮 */}
          <button
            onClick={handleBatchExport}
            className="flex items-center gap-1 rounded-lg border px-3 py-2 text-sm hover:bg-muted"
          >
            <Download className="h-4 w-4" />
            导出
          </button>

          {/* 清理按钮 */}
          <button
            onClick={() => setCleanupDialogOpen(true)}
            className="flex items-center gap-1 rounded-lg border px-3 py-2 text-sm hover:bg-muted"
            title="清理日志数据"
          >
            <Trash2 className="h-4 w-4 text-red-500" />
            清理
          </button>
        </div>
      </div>

      {/* 详情视图 */}
      {selectedFlow ? (
        <FlowDetail
          flowId={selectedFlow.id}
          onBack={handleBackToList}
          onExport={handleExportFlow}
        />
      ) : (
        <>
          {/* 过滤器 */}
          <FlowFilters filter={filter} onChange={setFilter} />

          {/* 主内容区域 */}
          {viewMode === "list" ? (
            <FlowList
              key={refreshKey}
              filter={filter}
              onFlowSelect={handleFlowSelect}
              enableRealtime={true}
            />
          ) : (
            <FlowStats
              key={refreshKey}
              filter={filter}
              autoRefreshInterval={30000}
            />
          )}
        </>
      )}

      {/* 导出对话框 */}
      <ExportDialog
        open={exportDialogOpen}
        onClose={() => setExportDialogOpen(false)}
        flowIds={exportFlowIds.length > 0 ? exportFlowIds : undefined}
        filter={exportFlowIds.length === 0 ? filter : undefined}
        onExportSuccess={handleExportSuccess}
      />

      {/* 清理对话框 */}
      <CleanupDialog
        isOpen={cleanupDialogOpen}
        onClose={() => setCleanupDialogOpen(false)}
        onSuccess={handleCleanupSuccess}
      />
    </div>
  );
}

export default FlowMonitorPage;
