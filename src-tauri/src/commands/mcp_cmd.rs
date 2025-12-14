use crate::database::DbConnection;
use crate::models::McpServer;
use crate::services::mcp_service::McpService;
use tauri::State;

#[tauri::command]
pub fn get_mcp_servers(db: State<'_, DbConnection>) -> Result<Vec<McpServer>, String> {
    McpService::get_all(&db)
}

#[tauri::command]
pub fn add_mcp_server(db: State<'_, DbConnection>, server: McpServer) -> Result<(), String> {
    McpService::add(&db, server)
}

#[tauri::command]
pub fn update_mcp_server(db: State<'_, DbConnection>, server: McpServer) -> Result<(), String> {
    McpService::update(&db, server)
}

#[tauri::command]
pub fn delete_mcp_server(db: State<'_, DbConnection>, id: String) -> Result<(), String> {
    McpService::delete(&db, &id)
}

#[tauri::command]
pub fn toggle_mcp_server(
    db: State<'_, DbConnection>,
    id: String,
    app_type: String,
    enabled: bool,
) -> Result<(), String> {
    McpService::toggle_enabled(&db, &id, &app_type, enabled)
}

#[tauri::command]
pub fn import_mcp_from_app(db: State<'_, DbConnection>, app_type: String) -> Result<usize, String> {
    McpService::import_from_app(&db, &app_type)
}

#[tauri::command]
pub fn sync_all_mcp_to_live(db: State<'_, DbConnection>) -> Result<(), String> {
    McpService::sync_all_to_live(&db)
}
