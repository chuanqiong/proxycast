use crate::models::McpServer;
use rusqlite::{params, Connection};
use serde_json::Value;

pub struct McpDao;

impl McpDao {
    pub fn get_all(conn: &Connection) -> Result<Vec<McpServer>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, name, server_config, description, enabled_proxycast,
                    enabled_claude, enabled_codex, enabled_gemini, created_at
             FROM mcp_servers ORDER BY created_at",
        )?;

        let servers = stmt.query_map([], |row| {
            let config_str: String = row.get(2)?;
            let server_config: Value = serde_json::from_str(&config_str).unwrap_or(Value::Null);

            Ok(McpServer {
                id: row.get(0)?,
                name: row.get(1)?,
                server_config,
                description: row.get(3)?,
                enabled_proxycast: row.get::<_, i32>(4)? == 1,
                enabled_claude: row.get::<_, i32>(5)? == 1,
                enabled_codex: row.get::<_, i32>(6)? == 1,
                enabled_gemini: row.get::<_, i32>(7)? == 1,
                created_at: row.get(8)?,
            })
        })?;

        servers.collect()
    }

    pub fn insert(conn: &Connection, server: &McpServer) -> Result<(), rusqlite::Error> {
        conn.execute(
            "INSERT INTO mcp_servers (id, name, server_config, description,
                                     enabled_proxycast, enabled_claude, enabled_codex,
                                     enabled_gemini, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                server.id,
                server.name,
                serde_json::to_string(&server.server_config).unwrap_or_default(),
                server.description,
                if server.enabled_proxycast { 1 } else { 0 },
                if server.enabled_claude { 1 } else { 0 },
                if server.enabled_codex { 1 } else { 0 },
                if server.enabled_gemini { 1 } else { 0 },
                server.created_at,
            ],
        )?;
        Ok(())
    }

    pub fn update(conn: &Connection, server: &McpServer) -> Result<(), rusqlite::Error> {
        conn.execute(
            "UPDATE mcp_servers SET name = ?1, server_config = ?2, description = ?3,
             enabled_proxycast = ?4, enabled_claude = ?5, enabled_codex = ?6, enabled_gemini = ?7
             WHERE id = ?8",
            params![
                server.name,
                serde_json::to_string(&server.server_config).unwrap_or_default(),
                server.description,
                if server.enabled_proxycast { 1 } else { 0 },
                if server.enabled_claude { 1 } else { 0 },
                if server.enabled_codex { 1 } else { 0 },
                if server.enabled_gemini { 1 } else { 0 },
                server.id,
            ],
        )?;
        Ok(())
    }

    pub fn delete(conn: &Connection, id: &str) -> Result<(), rusqlite::Error> {
        conn.execute("DELETE FROM mcp_servers WHERE id = ?", [id])?;
        Ok(())
    }

    pub fn toggle_enabled(
        conn: &Connection,
        id: &str,
        app_type: &str,
        enabled: bool,
    ) -> Result<(), rusqlite::Error> {
        let column = match app_type {
            "proxycast" => "enabled_proxycast",
            "claude" => "enabled_claude",
            "codex" => "enabled_codex",
            "gemini" => "enabled_gemini",
            _ => {
                return Err(rusqlite::Error::InvalidParameterName(format!(
                    "Invalid app_type: {}",
                    app_type
                )))
            }
        };

        let sql = format!("UPDATE mcp_servers SET {} = ? WHERE id = ?", column);
        conn.execute(&sql, params![if enabled { 1 } else { 0 }, id])?;
        Ok(())
    }
}
