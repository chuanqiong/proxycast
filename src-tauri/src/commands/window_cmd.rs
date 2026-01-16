//! 窗口控制命令
//!
//! 提供窗口大小调整、位置控制等功能

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, PhysicalSize};

/// 窗口大小预设
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowSize {
    pub width: u32,
    pub height: u32,
}

/// 预定义的窗口大小
impl WindowSize {
    /// 默认窗口大小
    pub fn default() -> Self {
        Self {
            width: 1200,
            height: 800,
        }
    }

    /// 紧凑模式
    pub fn compact() -> Self {
        Self {
            width: 1000,
            height: 700,
        }
    }

    /// 大屏模式
    pub fn large() -> Self {
        Self {
            width: 1920,
            height: 1200,
        }
    }

    /// 超大屏模式
    pub fn extra_large() -> Self {
        Self {
            width: 2560,
            height: 1440,
        }
    }
}

/// 窗口大小选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowSizeOption {
    pub id: String,
    pub name: String,
    pub description: String,
    pub size: WindowSize,
}

impl WindowSizeOption {
    /// 获取所有可用的窗口大小选项
    pub fn all_options() -> Vec<Self> {
        vec![
            Self {
                id: "compact".to_string(),
                name: "紧凑模式".to_string(),
                description: "1000×700 - 节省屏幕空间".to_string(),
                size: WindowSize::compact(),
            },
            Self {
                id: "default".to_string(),
                name: "默认大小".to_string(),
                description: "1200×800 - 日常使用".to_string(),
                size: WindowSize::default(),
            },
            Self {
                id: "large".to_string(),
                name: "大屏模式".to_string(),
                description: "1920×1200 - 大屏幕显示".to_string(),
                size: WindowSize::large(),
            },
            Self {
                id: "extra_large".to_string(),
                name: "超大屏模式".to_string(),
                description: "2560×1440 - 超大屏幕".to_string(),
                size: WindowSize::extra_large(),
            },
        ]
    }
}

/// 获取所有可用的窗口大小选项
///
/// # Returns
/// * `Vec<WindowSizeOption>` - 所有可用的窗口大小选项
#[tauri::command]
pub async fn get_window_size_options() -> Vec<WindowSizeOption> {
    WindowSizeOption::all_options()
}

/// 设置窗口为指定的预设大小
///
/// # Arguments
/// * `app` - Tauri AppHandle
/// * `option_id` - 窗口大小选项 ID
///
/// # Returns
/// * `Ok(WindowSize)` - 成功时返回之前的窗口大小
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn set_window_size_by_option(
    app: AppHandle,
    option_id: String,
) -> Result<WindowSize, String> {
    // 获取当前大小
    let current_size = get_window_size(app.clone()).await?;

    // 查找对应的窗口大小选项
    let options = WindowSizeOption::all_options();
    let option = options
        .iter()
        .find(|opt| opt.id == option_id)
        .ok_or_else(|| format!("未找到窗口大小选项: {}", option_id))?;

    // 设置新的窗口大小
    set_window_size(app.clone(), option.size.clone()).await?;

    // 居中窗口
    center_window(app).await?;

    Ok(current_size)
}

/// 切换全屏模式
///
/// # Arguments
/// * `app` - Tauri AppHandle
///
/// # Returns
/// * `Ok(bool)` - 成功时返回是否进入了全屏模式
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn toggle_fullscreen(app: AppHandle) -> Result<bool, String> {
    let window = app.get_webview_window("main").ok_or("无法获取主窗口")?;

    let is_fullscreen = window
        .is_fullscreen()
        .map_err(|e| format!("获取全屏状态失败: {}", e))?;

    window
        .set_fullscreen(!is_fullscreen)
        .map_err(|e| format!("切换全屏模式失败: {}", e))?;

    Ok(!is_fullscreen)
}

/// 检查是否处于全屏模式
///
/// # Arguments
/// * `app` - Tauri AppHandle
///
/// # Returns
/// * `Ok(bool)` - 成功时返回是否处于全屏模式
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn is_fullscreen(app: AppHandle) -> Result<bool, String> {
    let window = app.get_webview_window("main").ok_or("无法获取主窗口")?;

    window
        .is_fullscreen()
        .map_err(|e| format!("获取全屏状态失败: {}", e))
}

/// 获取当前窗口大小
///
/// # Arguments
/// * `app` - Tauri AppHandle
///
/// # Returns
/// * `Ok(WindowSize)` - 成功时返回当前窗口大小
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn get_window_size(app: AppHandle) -> Result<WindowSize, String> {
    let window = app.get_webview_window("main").ok_or("无法获取主窗口")?;

    let size = window
        .inner_size()
        .map_err(|e| format!("获取窗口大小失败: {}", e))?;

    Ok(WindowSize {
        width: size.width,
        height: size.height,
    })
}

/// 设置窗口大小
///
/// # Arguments
/// * `app` - Tauri AppHandle
/// * `size` - 新的窗口大小
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn set_window_size(app: AppHandle, size: WindowSize) -> Result<(), String> {
    let window = app.get_webview_window("main").ok_or("无法获取主窗口")?;

    let physical_size = PhysicalSize::new(size.width, size.height);

    window
        .set_size(physical_size)
        .map_err(|e| format!("设置窗口大小失败: {}", e))?;

    Ok(())
}

/// 恢复窗口到指定大小
///
/// # Arguments
/// * `app` - Tauri AppHandle
/// * `size` - 要恢复的窗口大小
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn restore_window_size(app: AppHandle, size: WindowSize) -> Result<(), String> {
    set_window_size(app, size).await
}

/// 居中窗口
///
/// # Arguments
/// * `app` - Tauri AppHandle
///
/// # Returns
/// * `Ok(())` - 成功
/// * `Err(String)` - 失败时返回错误消息
#[tauri::command]
pub async fn center_window(app: AppHandle) -> Result<(), String> {
    let window = app.get_webview_window("main").ok_or("无法获取主窗口")?;

    window
        .center()
        .map_err(|e| format!("居中窗口失败: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_size_presets() {
        let default = WindowSize::default();
        assert_eq!(default.width, 1200);
        assert_eq!(default.height, 800);

        let compact = WindowSize::compact();
        assert_eq!(compact.width, 1000);
        assert_eq!(compact.height, 700);

        let large = WindowSize::large();
        assert_eq!(large.width, 1920);
        assert_eq!(large.height, 1200);

        let extra_large = WindowSize::extra_large();
        assert_eq!(extra_large.width, 2560);
        assert_eq!(extra_large.height, 1440);
    }

    #[test]
    fn test_window_size_options() {
        let options = WindowSizeOption::all_options();
        assert_eq!(options.len(), 4);

        // 验证每个选项都有有效的 ID 和名称
        for option in &options {
            assert!(!option.id.is_empty());
            assert!(!option.name.is_empty());
            assert!(!option.description.is_empty());
            assert!(option.size.width > 0);
            assert!(option.size.height > 0);
        }

        // 验证特定选项
        let default_option = options.iter().find(|opt| opt.id == "default").unwrap();
        assert_eq!(default_option.size.width, 1200);
        assert_eq!(default_option.size.height, 800);
    }
}
