use crate::database::dao::providers::ProviderDao;
use crate::database::DbConnection;
use crate::models::{AppType, Provider};
use crate::services::live_sync;

pub struct SwitchService;

impl SwitchService {
    pub fn get_providers(db: &DbConnection, app_type: &str) -> Result<Vec<Provider>, String> {
        let conn = db.lock().map_err(|e| e.to_string())?;
        ProviderDao::get_all(&conn, app_type).map_err(|e| e.to_string())
    }

    pub fn get_current_provider(
        db: &DbConnection,
        app_type: &str,
    ) -> Result<Option<Provider>, String> {
        let conn = db.lock().map_err(|e| e.to_string())?;
        ProviderDao::get_current(&conn, app_type).map_err(|e| e.to_string())
    }

    pub fn add_provider(db: &DbConnection, provider: Provider) -> Result<(), String> {
        let conn = db.lock().map_err(|e| e.to_string())?;

        // Check if this is the first provider for this app type
        let existing = ProviderDao::get_all(&conn, &provider.app_type).map_err(|e| e.to_string())?;
        let is_first = existing.is_empty();

        ProviderDao::insert(&conn, &provider).map_err(|e| e.to_string())?;

        // If this is the first provider, automatically set it as current and sync
        if is_first {
            ProviderDao::set_current(&conn, &provider.app_type, &provider.id)
                .map_err(|e| e.to_string())?;

            if let Ok(app_type_enum) = provider.app_type.parse::<AppType>() {
                if app_type_enum != AppType::ProxyCast {
                    live_sync::sync_to_live(&app_type_enum, &provider)
                        .map_err(|e| format!("Failed to sync: {}", e))?;
                }
            }
        }

        Ok(())
    }

    pub fn update_provider(db: &DbConnection, provider: Provider) -> Result<(), String> {
        let conn = db.lock().map_err(|e| e.to_string())?;

        // Check if this is the current provider
        let current = ProviderDao::get_current(&conn, &provider.app_type).map_err(|e| e.to_string())?;
        let is_current = current.as_ref().map(|p| p.id == provider.id).unwrap_or(false);

        ProviderDao::update(&conn, &provider).map_err(|e| e.to_string())?;

        // If this is the current provider, sync to live
        if is_current {
            if let Ok(app_type_enum) = provider.app_type.parse::<AppType>() {
                if app_type_enum != AppType::ProxyCast {
                    live_sync::sync_to_live(&app_type_enum, &provider)
                        .map_err(|e| format!("Failed to sync: {}", e))?;
                }
            }
        }

        Ok(())
    }

    pub fn delete_provider(db: &DbConnection, app_type: &str, id: &str) -> Result<(), String> {
        let conn = db.lock().map_err(|e| e.to_string())?;

        // Check if trying to delete the current provider
        let current = ProviderDao::get_current(&conn, app_type).map_err(|e| e.to_string())?;
        if let Some(ref current_provider) = current {
            if current_provider.id == id {
                return Err("Cannot delete the currently active provider".to_string());
            }
        }

        ProviderDao::delete(&conn, app_type, id).map_err(|e| e.to_string())
    }

    pub fn switch_provider(db: &DbConnection, app_type: &str, id: &str) -> Result<(), String> {
        let conn = db.lock().map_err(|e| e.to_string())?;

        // Get target provider
        let target_provider = ProviderDao::get_by_id(&conn, app_type, id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Provider not found: {}", id))?;

        let app_type_enum = app_type.parse::<AppType>().map_err(|e| e.to_string())?;

        // Skip backfill and sync for ProxyCast
        if app_type_enum != AppType::ProxyCast {
            // Backfill: Read current live config and save to current provider
            if let Some(current_provider) = ProviderDao::get_current(&conn, app_type)
                .map_err(|e| e.to_string())?
            {
                // Only backfill if switching to a different provider
                if current_provider.id != id {
                    if let Ok(live_settings) = live_sync::read_live_settings(&app_type_enum) {
                        // Update current provider with live settings
                        let mut updated_provider = current_provider.clone();
                        updated_provider.settings_config = live_settings;
                        let _ = ProviderDao::update(&conn, &updated_provider);
                    }
                }
            }
        }

        // Set new current provider
        ProviderDao::set_current(&conn, app_type, id).map_err(|e| e.to_string())?;

        // Sync target provider to live config
        if app_type_enum != AppType::ProxyCast {
            live_sync::sync_to_live(&app_type_enum, &target_provider)
                .map_err(|e| format!("Failed to sync: {}", e))?;
        }

        Ok(())
    }

    /// Import current live config as a default provider
    pub fn import_default_config(db: &DbConnection, app_type: &str) -> Result<bool, String> {
        let conn = db.lock().map_err(|e| e.to_string())?;

        // Check if providers already exist
        let existing = ProviderDao::get_all(&conn, app_type).map_err(|e| e.to_string())?;
        if !existing.is_empty() {
            return Ok(false); // Already has providers, skip import
        }

        let app_type_enum = app_type.parse::<AppType>().map_err(|e| e.to_string())?;

        // Skip for ProxyCast
        if app_type_enum == AppType::ProxyCast {
            return Ok(false);
        }

        // Read live settings
        let live_settings = live_sync::read_live_settings(&app_type_enum)
            .map_err(|e| format!("Failed to read live settings: {}", e))?;

        // Create default provider
        let provider = Provider {
            id: "default".to_string(),
            app_type: app_type.to_string(),
            name: "Default (Imported)".to_string(),
            settings_config: live_settings,
            category: Some("custom".to_string()),
            icon: None,
            icon_color: Some("#6366f1".to_string()),
            notes: Some("Imported from existing configuration".to_string()),
            is_current: true,
            sort_index: Some(0),
            created_at: Some(chrono::Utc::now().timestamp()),
        };

        ProviderDao::insert(&conn, &provider).map_err(|e| e.to_string())?;

        Ok(true)
    }

    /// Read current live settings for an app type
    pub fn read_live_settings(app_type: &str) -> Result<serde_json::Value, String> {
        let app_type_enum = app_type.parse::<AppType>().map_err(|e| e.to_string())?;
        live_sync::read_live_settings(&app_type_enum).map_err(|e| e.to_string())
    }
}
