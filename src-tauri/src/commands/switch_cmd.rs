use crate::database::DbConnection;
use crate::models::Provider;
use crate::services::switch::SwitchService;
use serde_json::Value;
use tauri::State;

#[tauri::command]
pub fn get_switch_providers(
    db: State<'_, DbConnection>,
    app_type: String,
) -> Result<Vec<Provider>, String> {
    SwitchService::get_providers(&db, &app_type)
}

#[tauri::command]
pub fn get_current_switch_provider(
    db: State<'_, DbConnection>,
    app_type: String,
) -> Result<Option<Provider>, String> {
    SwitchService::get_current_provider(&db, &app_type)
}

#[tauri::command]
pub fn add_switch_provider(db: State<'_, DbConnection>, provider: Provider) -> Result<(), String> {
    SwitchService::add_provider(&db, provider)
}

#[tauri::command]
pub fn update_switch_provider(
    db: State<'_, DbConnection>,
    provider: Provider,
) -> Result<(), String> {
    SwitchService::update_provider(&db, provider)
}

#[tauri::command]
pub fn delete_switch_provider(
    db: State<'_, DbConnection>,
    app_type: String,
    id: String,
) -> Result<(), String> {
    SwitchService::delete_provider(&db, &app_type, &id)
}

#[tauri::command]
pub fn switch_provider(
    db: State<'_, DbConnection>,
    app_type: String,
    id: String,
) -> Result<(), String> {
    SwitchService::switch_provider(&db, &app_type, &id)
}

#[tauri::command]
pub fn import_default_config(
    db: State<'_, DbConnection>,
    app_type: String,
) -> Result<bool, String> {
    SwitchService::import_default_config(&db, &app_type)
}

#[tauri::command]
pub fn read_live_provider_settings(app_type: String) -> Result<Value, String> {
    SwitchService::read_live_settings(&app_type)
}
