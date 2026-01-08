//! 配置管理命令
//!
//! 包含配置读取、保存、Provider 设置等命令。

use crate::app::types::{AppState, LogState};
use crate::app::utils::{is_non_local_bind, is_valid_bind_host};
use crate::config::{
    self,
    observer::{ConfigChangeEvent, RoutingChangeEvent},
    ConfigChangeSource, GlobalConfigManagerState, DEFAULT_API_KEY,
};

/// 获取配置
#[tauri::command]
pub async fn get_config(state: tauri::State<'_, AppState>) -> Result<config::Config, String> {
    let s = state.read().await;
    Ok(s.config.clone())
}

/// 保存配置
#[tauri::command]
pub async fn save_config(
    state: tauri::State<'_, AppState>,
    config: config::Config,
) -> Result<(), String> {
    let host = config.server.host.to_lowercase();

    // 验证绑定地址
    if !is_valid_bind_host(&host) {
        return Err(
            "无效的监听地址。允许的地址：127.0.0.1、localhost、::1、0.0.0.0、::".to_string(),
        );
    }

    // 如果监听所有接口，要求使用强 API Key
    if is_non_local_bind(&host) && config.server.api_key == DEFAULT_API_KEY {
        return Err(
            "安全限制：监听所有网络接口 (0.0.0.0 或 ::) 时，必须设置非默认的 API Key".to_string(),
        );
    }

    // 禁止开启远程管理
    if config.remote_management.allow_remote {
        return Err("安全限制：不允许开启远程管理功能".to_string());
    }

    let mut s = state.write().await;
    s.config = config.clone();
    config::save_config(&config).map_err(|e| e.to_string())
}

/// 获取默认 Provider
#[tauri::command]
pub async fn get_default_provider(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let s = state.read().await;
    Ok(s.config.default_provider.clone())
}

/// 设置默认 Provider
#[tauri::command]
pub async fn set_default_provider(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    config_manager: tauri::State<'_, GlobalConfigManagerState>,
    provider: String,
) -> Result<String, String> {
    // 更新 AppState 中的配置
    let mut s = state.write().await;
    s.config.default_provider = provider.clone();
    s.config.routing.default_provider = provider.clone();

    // 同时更新运行中服务器的 default_provider_ref（向后兼容）
    {
        let mut dp = s.default_provider_ref.write().await;
        *dp = provider.clone();
    }

    // 同时更新运行中服务器的 router（如果服务器正在运行）
    if let Some(router_ref) = &s.router_ref {
        if let Ok(provider_type) = provider.parse::<crate::ProviderType>() {
            let mut router = router_ref.write().await;
            router.set_default_provider(provider_type);
        }
    }

    // 保存配置
    config::save_config(&s.config).map_err(|e| e.to_string())?;

    // 释放锁后通知观察者
    drop(s);

    // 通过 GlobalConfigManager 通知所有观察者
    let event = ConfigChangeEvent::RoutingChanged(RoutingChangeEvent {
        default_provider: Some(provider.clone()),
        model_aliases_changed: false,
        model_aliases: None,
        source: ConfigChangeSource::FrontendUI,
    });
    config_manager.0.subject().notify_event(event).await;

    logs.write()
        .await
        .add("info", &format!("默认 Provider 已切换为: {provider}"));

    tracing::info!("[CONFIG] 默认 Provider 已更新: {}", provider);
    Ok(provider)
}

/// 获取端点 Provider 配置
#[tauri::command]
pub async fn get_endpoint_providers(
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let s = state.read().await;
    let ep = &s.config.endpoint_providers;
    Ok(serde_json::json!({
        "cursor": ep.cursor.clone(),
        "claude_code": ep.claude_code.clone(),
        "codex": ep.codex.clone(),
        "windsurf": ep.windsurf.clone(),
        "kiro": ep.kiro.clone(),
        "other": ep.other.clone()
    }))
}

/// 设置端点 Provider 配置
#[tauri::command]
pub async fn set_endpoint_provider(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    config_manager: tauri::State<'_, GlobalConfigManagerState>,
    endpoint: String,
    provider: Option<String>,
) -> Result<String, String> {
    // 允许任意 Provider ID（包括自定义 Provider 的 UUID）
    // 不再强制验证为已知的 ProviderType

    let ep_config = {
        let mut s = state.write().await;

        // 使用 set_provider 方法设置对应的 provider
        if !s
            .config
            .endpoint_providers
            .set_provider(&endpoint, provider.clone())
        {
            return Err(format!("未知的客户端类型: {}", endpoint));
        }

        config::save_config(&s.config).map_err(|e| e.to_string())?;

        s.config.endpoint_providers.clone()
    };

    // 通过 GlobalConfigManager 通知所有观察者
    let event = ConfigChangeEvent::EndpointProvidersChanged(
        config::observer::EndpointProvidersChangeEvent {
            cursor: ep_config.cursor.clone(),
            claude_code: ep_config.claude_code.clone(),
            codex: ep_config.codex.clone(),
            windsurf: ep_config.windsurf.clone(),
            kiro: ep_config.kiro.clone(),
            other: ep_config.other.clone(),
            source: ConfigChangeSource::FrontendUI,
        },
    );
    config_manager.0.subject().notify_event(event).await;

    let provider_display = provider.as_deref().unwrap_or("默认");
    logs.write().await.add(
        "info",
        &format!(
            "客户端 {} 的 Provider 已设置为: {}",
            endpoint, provider_display
        ),
    );

    tracing::info!(
        "[CONFIG] 端点 Provider 已更新: {} -> {}",
        endpoint,
        provider_display
    );
    Ok(provider_display.to_string())
}
