use crate::models::AppType;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::AppHandle;
use tauri_plugin_autostart::ManagerExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigStatus {
    pub exists: bool,
    pub path: String,
    pub has_env: bool,
}

/// Get the config directory path for an app type
fn get_config_dir(app_type: &AppType) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    match app_type {
        AppType::Claude => Some(home.join(".claude")),
        AppType::Codex => Some(home.join(".codex")),
        AppType::Gemini => Some(home.join(".gemini")),
        AppType::ProxyCast => dirs::config_dir().map(|d| d.join("proxycast")),
    }
}

#[tauri::command]
pub fn get_config_status(app_type: String) -> Result<ConfigStatus, String> {
    let app = app_type.parse::<AppType>().map_err(|e| e.to_string())?;
    let config_dir = get_config_dir(&app).ok_or("Cannot determine config directory")?;

    let main_config = match app {
        AppType::Claude => config_dir.join("settings.json"),
        AppType::Codex => config_dir.join("auth.json"),
        AppType::Gemini => config_dir.join(".env"),
        AppType::ProxyCast => config_dir.join("config.json"),
    };

    let has_env = match app {
        AppType::Claude => {
            config_dir.join("settings.json").exists()
                && std::fs::read_to_string(config_dir.join("settings.json"))
                    .map(|s| s.contains("env"))
                    .unwrap_or(false)
        }
        AppType::Codex => config_dir.join("auth.json").exists(),
        AppType::Gemini => config_dir.join(".env").exists(),
        AppType::ProxyCast => config_dir.join("config.json").exists(),
    };

    Ok(ConfigStatus {
        exists: main_config.exists(),
        path: config_dir.to_string_lossy().to_string(),
        has_env,
    })
}

#[tauri::command]
pub fn get_config_dir_path(app_type: String) -> Result<String, String> {
    let app = app_type.parse::<AppType>().map_err(|e| e.to_string())?;
    let config_dir = get_config_dir(&app).ok_or("Cannot determine config directory")?;
    Ok(config_dir.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn open_config_folder(handle: AppHandle, app_type: String) -> Result<bool, String> {
    let app = app_type.parse::<AppType>().map_err(|e| e.to_string())?;
    let config_dir = get_config_dir(&app).ok_or("Cannot determine config directory")?;

    if !config_dir.exists() {
        std::fs::create_dir_all(&config_dir).map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&config_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(&config_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&config_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    Ok(true)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolVersion {
    pub name: String,
    pub version: Option<String>,
    pub installed: bool,
}

#[tauri::command]
pub async fn get_tool_versions() -> Result<Vec<ToolVersion>, String> {
    let mut versions = Vec::new();

    // Check Claude Code version
    let claude_version = std::process::Command::new("claude")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string());

    versions.push(ToolVersion {
        name: "Claude Code".to_string(),
        version: claude_version.clone(),
        installed: claude_version.is_some(),
    });

    // Check Codex version
    let codex_version = std::process::Command::new("codex")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string());

    versions.push(ToolVersion {
        name: "Codex".to_string(),
        version: codex_version.clone(),
        installed: codex_version.is_some(),
    });

    // Check Gemini CLI version
    let gemini_version = std::process::Command::new("gemini")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string());

    versions.push(ToolVersion {
        name: "Gemini CLI".to_string(),
        version: gemini_version.clone(),
        installed: gemini_version.is_some(),
    });

    Ok(versions)
}

#[tauri::command]
pub async fn get_auto_launch_status(app: AppHandle) -> Result<bool, String> {
    let autostart_manager = app.autolaunch();
    autostart_manager
        .is_enabled()
        .map_err(|e| format!("Failed to get autostart status: {}", e))
}

#[tauri::command]
pub async fn set_auto_launch(app: AppHandle, enabled: bool) -> Result<bool, String> {
    let autostart_manager = app.autolaunch();

    if enabled {
        autostart_manager
            .enable()
            .map_err(|e| format!("Failed to enable autostart: {}", e))?;
    } else {
        autostart_manager
            .disable()
            .map_err(|e| format!("Failed to disable autostart: {}", e))?;
    }

    Ok(enabled)
}
