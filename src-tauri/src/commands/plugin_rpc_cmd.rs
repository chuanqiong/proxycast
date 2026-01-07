//! 插件 RPC 通信命令
//!
//! 提供插件与其 Binary 后端进程的 JSON-RPC 通信功能：
//! - plugin_rpc_connect: 启动插件进程并建立连接
//! - plugin_rpc_disconnect: 关闭插件进程
//! - plugin_rpc_call: 发送 RPC 请求并等待响应
//!
//! _需求: 插件 RPC 通信_

use crate::commands::plugin_install_cmd::PluginInstallerState;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// RPC 请求 ID 生成器
static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// JSON-RPC 请求
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    method: String,
    params: Option<Value>,
    id: u64,
}

/// JSON-RPC 响应
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    result: Option<Value>,
    error: Option<JsonRpcError>,
    #[allow(dead_code)]
    id: Option<u64>,
}

/// JSON-RPC 错误
#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[allow(dead_code)]
    data: Option<Value>,
}

/// 插件进程信息
struct PluginProcess {
    child: Child,
    #[allow(dead_code)]
    plugin_id: String,
}

/// 插件 RPC 管理器状态
pub struct PluginRpcManagerState {
    /// 运行中的插件进程
    processes: RwLock<HashMap<String, Arc<Mutex<PluginProcess>>>>,
}

impl PluginRpcManagerState {
    pub fn new() -> Self {
        Self {
            processes: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for PluginRpcManagerState {
    fn default() -> Self {
        Self::new()
    }
}

/// 启动插件进程并建立 RPC 连接
#[tauri::command]
pub async fn plugin_rpc_connect(
    plugin_id: String,
    installer_state: tauri::State<'_, PluginInstallerState>,
    rpc_state: tauri::State<'_, PluginRpcManagerState>,
) -> Result<(), String> {
    // 检查是否已连接
    {
        let processes = rpc_state.processes.read().await;
        if processes.contains_key(&plugin_id) {
            return Ok(()); // 已连接
        }
    }

    // 获取插件信息
    let installer = installer_state.0.read().await;
    let plugins = installer.list_installed().map_err(|e| e.to_string())?;
    let plugin = plugins
        .iter()
        .find(|p| p.id == plugin_id)
        .ok_or_else(|| format!("插件 {} 未安装", plugin_id))?;

    // 读取插件 manifest
    let manifest_path = plugin.install_path.join("plugin.json");
    let manifest_content = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("读取 manifest 失败: {}", e))?;
    let manifest: Value = serde_json::from_str(&manifest_content)
        .map_err(|e| format!("解析 manifest 失败: {}", e))?;

    // 获取二进制文件路径
    let binary_name = manifest["binary"]["binary_name"]
        .as_str()
        .ok_or("manifest 中缺少 binary.binary_name")?;

    // 根据平台选择二进制文件
    let platform_key = match (std::env::consts::ARCH, std::env::consts::OS) {
        ("aarch64", "macos") => "macos-arm64",
        ("x86_64", "macos") => "macos-x64",
        ("x86_64", "linux") => "linux-x64",
        ("aarch64", "linux") => "linux-arm64",
        ("x86_64", "windows") => "windows-x64",
        _ => return Err("不支持的平台".to_string()),
    };

    let binary_filename = manifest["binary"]["platform_binaries"][platform_key]
        .as_str()
        .ok_or_else(|| format!("manifest 中缺少 {} 平台的二进制文件", platform_key))?;

    let binary_path = plugin.install_path.join(binary_filename);
    if !binary_path.exists() {
        return Err(format!("二进制文件不存在: {:?}", binary_path));
    }

    // 启动进程
    let child = Command::new(&binary_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("启动插件进程失败: {}", e))?;

    tracing::info!("插件 {} 进程已启动, PID: {:?}", plugin_id, child.id());

    let process = PluginProcess {
        child,
        plugin_id: plugin_id.clone(),
    };

    // 保存进程
    let mut processes = rpc_state.processes.write().await;
    processes.insert(plugin_id, Arc::new(Mutex::new(process)));

    Ok(())
}

/// 关闭插件 RPC 连接
#[tauri::command]
pub async fn plugin_rpc_disconnect(
    plugin_id: String,
    rpc_state: tauri::State<'_, PluginRpcManagerState>,
) -> Result<(), String> {
    let mut processes = rpc_state.processes.write().await;

    if let Some(process_arc) = processes.remove(&plugin_id) {
        let mut process = process_arc.lock().await;
        if let Err(e) = process.child.kill() {
            tracing::warn!("关闭插件 {} 进程失败: {}", plugin_id, e);
        }
        tracing::info!("插件 {} 进程已关闭", plugin_id);
    }

    Ok(())
}

/// 发送 RPC 请求
#[tauri::command]
pub async fn plugin_rpc_call(
    plugin_id: String,
    method: String,
    params: Option<Value>,
    rpc_state: tauri::State<'_, PluginRpcManagerState>,
) -> Result<Value, String> {
    let processes = rpc_state.processes.read().await;
    let process_arc = processes
        .get(&plugin_id)
        .ok_or_else(|| format!("插件 {} 未连接", plugin_id))?
        .clone();
    drop(processes);

    let mut process = process_arc.lock().await;

    // 构建请求
    let request_id = REQUEST_ID.fetch_add(1, Ordering::SeqCst);
    let request = JsonRpcRequest {
        jsonrpc: "2.0",
        method,
        params,
        id: request_id,
    };

    let request_json =
        serde_json::to_string(&request).map_err(|e| format!("序列化请求失败: {}", e))?;

    // 发送请求
    let stdin = process.child.stdin.as_mut().ok_or("无法获取进程 stdin")?;
    writeln!(stdin, "{}", request_json).map_err(|e| format!("发送请求失败: {}", e))?;
    stdin
        .flush()
        .map_err(|e| format!("刷新 stdin 失败: {}", e))?;

    // 读取响应
    let stdout = process.child.stdout.as_mut().ok_or("无法获取进程 stdout")?;
    let mut reader = BufReader::new(stdout);
    let mut response_line = String::new();
    reader
        .read_line(&mut response_line)
        .map_err(|e| format!("读取响应失败: {}", e))?;

    // 解析响应
    let response: JsonRpcResponse =
        serde_json::from_str(&response_line).map_err(|e| format!("解析响应失败: {}", e))?;

    if let Some(error) = response.error {
        return Err(format!("RPC 错误 [{}]: {}", error.code, error.message));
    }

    Ok(response.result.unwrap_or(Value::Null))
}
