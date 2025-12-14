use crate::database::dao::mcp::McpDao;
use crate::database::DbConnection;
use crate::models::{AppType, McpServer};
use crate::services::mcp_sync;

pub struct McpService;

impl McpService {
    pub fn get_all(db: &DbConnection) -> Result<Vec<McpServer>, String> {
        let conn = db.lock().map_err(|e| e.to_string())?;
        McpDao::get_all(&conn).map_err(|e| e.to_string())
    }

    pub fn add(db: &DbConnection, server: McpServer) -> Result<(), String> {
        let conn = db.lock().map_err(|e| e.to_string())?;
        McpDao::insert(&conn, &server).map_err(|e| e.to_string())?;

        // Sync to enabled apps
        let servers = McpDao::get_all(&conn).map_err(|e| e.to_string())?;
        mcp_sync::sync_all_mcp_to_live(&servers).map_err(|e| e.to_string())?;

        Ok(())
    }

    pub fn update(db: &DbConnection, server: McpServer) -> Result<(), String> {
        let conn = db.lock().map_err(|e| e.to_string())?;
        McpDao::update(&conn, &server).map_err(|e| e.to_string())?;

        // Sync to enabled apps
        let servers = McpDao::get_all(&conn).map_err(|e| e.to_string())?;
        mcp_sync::sync_all_mcp_to_live(&servers).map_err(|e| e.to_string())?;

        Ok(())
    }

    pub fn delete(db: &DbConnection, id: &str) -> Result<(), String> {
        let conn = db.lock().map_err(|e| e.to_string())?;
        McpDao::delete(&conn, id).map_err(|e| e.to_string())?;

        // Remove from all apps
        mcp_sync::remove_mcp_from_all_apps(id).map_err(|e| e.to_string())?;

        Ok(())
    }

    pub fn toggle_enabled(
        db: &DbConnection,
        id: &str,
        app_type: &str,
        enabled: bool,
    ) -> Result<(), String> {
        let conn = db.lock().map_err(|e| e.to_string())?;
        McpDao::toggle_enabled(&conn, id, app_type, enabled).map_err(|e| e.to_string())?;

        // Get the server and sync
        let servers = McpDao::get_all(&conn).map_err(|e| e.to_string())?;
        let server = servers.iter().find(|s| s.id == id);

        if let Some(server) = server {
            let app = app_type.parse::<AppType>().map_err(|e| e.to_string())?;
            if enabled {
                // Sync server to the app
                mcp_sync::sync_mcp_to_app(&app, &servers).map_err(|e| e.to_string())?;
            } else {
                // Remove server from the app
                mcp_sync::remove_mcp_from_app(&app, id).map_err(|e| e.to_string())?;
            }
        }

        Ok(())
    }

    /// Sync all enabled MCP servers to all apps
    pub fn sync_all_to_live(db: &DbConnection) -> Result<(), String> {
        let conn = db.lock().map_err(|e| e.to_string())?;
        let servers = McpDao::get_all(&conn).map_err(|e| e.to_string())?;
        mcp_sync::sync_all_mcp_to_live(&servers).map_err(|e| e.to_string())
    }

    /// Import MCP servers from an app
    pub fn import_from_app(db: &DbConnection, app_type: &str) -> Result<usize, String> {
        let app = app_type.parse::<AppType>().map_err(|e| e.to_string())?;
        let conn = db.lock().map_err(|e| e.to_string())?;

        // Get existing servers
        let existing = McpDao::get_all(&conn).map_err(|e| e.to_string())?;
        let existing_ids: std::collections::HashSet<String> =
            existing.iter().map(|s| s.id.clone()).collect();

        // Import from app
        let imported = mcp_sync::import_mcp_from_app(&app).map_err(|e| e.to_string())?;

        let mut count = 0;
        for server in imported {
            if existing_ids.contains(&server.id) {
                // Update existing server's enabled status for this app
                if McpDao::toggle_enabled(&conn, &server.id, app_type, true).is_ok() {
                    count += 1;
                }
            } else {
                // Insert new server
                if McpDao::insert(&conn, &server).is_ok() {
                    count += 1;
                }
            }
        }

        Ok(count)
    }
}
