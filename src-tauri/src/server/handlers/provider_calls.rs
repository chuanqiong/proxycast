//! Provider 调用处理器
//!
//! 根据凭证类型调用不同的 Provider API

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::stream;

use crate::converter::anthropic_to_openai::convert_anthropic_to_openai;
use crate::converter::openai_to_antigravity::{
    convert_antigravity_to_openai_response, convert_openai_to_antigravity_with_context,
};
use crate::models::anthropic::AnthropicMessagesRequest;
use crate::models::openai::ChatCompletionRequest;
use crate::models::provider_pool_model::{CredentialData, ProviderCredential};
use crate::providers::{
    AntigravityProvider, ClaudeCustomProvider, GeminiProvider, KiroProvider, OpenAICustomProvider,
    QwenProvider, VertexProvider,
};
use crate::server::AppState;
use crate::server_utils::{
    build_anthropic_response, build_anthropic_stream_response, parse_cw_response, safe_truncate,
    CWParsedResponse,
};
/// 根据凭证调用 Provider (Anthropic 格式)
pub async fn call_provider_anthropic(
    state: &AppState,
    credential: &ProviderCredential,
    request: &AnthropicMessagesRequest,
) -> Response {
    match &credential.credential {
        CredentialData::KiroOAuth { creds_file_path } => {
            // 使用 TokenCacheService 获取有效 token
            let db = match &state.db {
                Some(db) => db,
                None => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": "Database not available"}})),
                    )
                        .into_response();
                }
            };
            // 获取缓存的 token
            let token = match state
                .token_cache
                .get_valid_token(db, &credential.uuid)
                .await
            {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("[POOL] Token cache miss, loading from source: {}", e);
                    // 回退到从源文件加载
                    let mut kiro = KiroProvider::new();
                    if let Err(e) = kiro.load_credentials_from_path(creds_file_path).await {
                        // 记录凭证加载失败
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Failed to load credentials: {}", e)),
                        );
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": format!("Failed to load Kiro credentials: {}", e)}})),
                        )
                            .into_response();
                    }
                    if let Err(e) = kiro.refresh_token().await {
                        // 记录 Token 刷新失败
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                        )
                            .into_response();
                    }
                    kiro.credentials.access_token.unwrap_or_default()
                }
            };
            // 使用获取到的 token 创建 KiroProvider
            let mut kiro = KiroProvider::new();
            kiro.credentials.access_token = Some(token);
            // 从源文件加载其他配置（region, profile_arn 等）
            let _ = kiro.load_credentials_from_path(creds_file_path).await;
            let openai_request = convert_anthropic_to_openai(request);
            let resp = match kiro.call_api(&openai_request).await {
                Ok(r) => r,
                Err(e) => {
                    // 记录 API 调用失败
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&e.to_string()),
                    );
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response();
                }
            };
            let status = resp.status();
            if status.is_success() {
                match resp.bytes().await {
                    Ok(bytes) => {
                        let body = String::from_utf8_lossy(&bytes).to_string();
                        let parsed = parse_cw_response(&body);
                        // 记录成功
                        let _ = state.pool_service.mark_healthy(
                            db,
                            &credential.uuid,
                            Some(&request.model),
                        );
                        let _ = state.pool_service.record_usage(db, &credential.uuid);
                        if request.stream {
                            build_anthropic_stream_response(&request.model, &parsed)
                        } else {
                            build_anthropic_response(&request.model, &parsed)
                        }
                    }
                    Err(e) => {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else if status.as_u16() == 401 || status.as_u16() == 403 {
                // Token 过期，强制刷新并重试
                tracing::info!(
                    "[POOL] Got {}, forcing token refresh for {}",
                    status,
                    &credential.uuid[..8]
                );
                let new_token = match state
                    .token_cache
                    .refresh_and_cache(db, &credential.uuid, true)
                    .await
                {
                    Ok(t) => t,
                    Err(e) => {
                        // 记录 Token 刷新失败
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                        )
                            .into_response();
                    }
                };
                // 使用新 token 重试
                kiro.credentials.access_token = Some(new_token);
                match kiro.call_api(&openai_request).await {
                    Ok(retry_resp) => {
                        if retry_resp.status().is_success() {
                            match retry_resp.bytes().await {
                                Ok(bytes) => {
                                    let body = String::from_utf8_lossy(&bytes).to_string();
                                    let parsed = parse_cw_response(&body);
                                    // 记录重试成功
                                    let _ = state.pool_service.mark_healthy(
                                        db,
                                        &credential.uuid,
                                        Some(&request.model),
                                    );
                                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                                    if request.stream {
                                        build_anthropic_stream_response(&request.model, &parsed)
                                    } else {
                                        build_anthropic_response(&request.model, &parsed)
                                    }
                                }
                                Err(e) => {
                                    let _ = state.pool_service.mark_unhealthy(
                                        db,
                                        &credential.uuid,
                                        Some(&e.to_string()),
                                    );
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                    )
                                        .into_response()
                                }
                            }
                        } else {
                            let body = retry_resp.text().await.unwrap_or_default();
                            let _ = state.pool_service.mark_unhealthy(
                                db,
                                &credential.uuid,
                                Some(&format!("Retry failed: {}", body)),
                            );
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": format!("Retry failed: {}", body)}})),
                            )
                                .into_response()
                        }
                    }
                    Err(e) => {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                let _ = state
                    .pool_service
                    .mark_unhealthy(db, &credential.uuid, Some(&body));
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": body}})),
                )
                    .into_response()
            }
        }
        CredentialData::GeminiOAuth { .. } => {
            // Gemini OAuth 路由暂不支持
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Gemini OAuth routing not yet implemented. Use /v1/messages with Gemini models instead."}})),
            )
                .into_response()
        }
        CredentialData::QwenOAuth { .. } => {
            // Qwen OAuth 路由暂不支持
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Qwen OAuth routing not yet implemented. Use /v1/messages with Qwen models instead."}})),
            )
                .into_response()
        }
        CredentialData::AntigravityOAuth {
            creds_file_path,
            project_id,
        } => {
            let mut antigravity = AntigravityProvider::new();
            if let Err(e) = antigravity
                .load_credentials_from_path(creds_file_path)
                .await
            {
                // 记录凭证加载失败
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&format!("Failed to load credentials: {}", e)),
                    );
                }
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": format!("Failed to load Antigravity credentials: {}", e)}})),
                )
                    .into_response();
            }
            // 检查并刷新 token
            if antigravity.is_token_expiring_soon() {
                if let Err(e) = antigravity.refresh_token().await {
                    // 记录 Token 刷新失败
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                    }
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                    )
                        .into_response();
                }
            }
            // 设置项目 ID
            if let Some(pid) = project_id {
                antigravity.project_id = Some(pid.clone());
            } else if let Err(e) = antigravity.discover_project().await {
                tracing::warn!("[Antigravity] Failed to discover project: {}", e);
            }
            // 获取 project_id 用于请求
            let proj_id = antigravity.project_id.clone().unwrap_or_default();
            // 先转换为 OpenAI 格式，再转换为 Antigravity 格式
            let openai_request = convert_anthropic_to_openai(request);
            let antigravity_request = convert_openai_to_antigravity_with_context(&openai_request, &proj_id);
            match antigravity
                .generate_content(&request.model, &antigravity_request)
                .await
            {
                Ok(resp) => {
                    // 转换为 OpenAI 格式，再构建 Anthropic 响应
                    let content = resp["candidates"][0]["content"]["parts"][0]["text"]
                        .as_str()
                        .unwrap_or("");
                    let parsed = CWParsedResponse {
                        content: content.to_string(),
                        tool_calls: Vec::new(),
                        usage_credits: 0.0,
                        context_usage_percentage: 0.0,
                    };
                    // 记录成功
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_healthy(
                            db,
                            &credential.uuid,
                            Some(&request.model),
                        );
                        let _ = state.pool_service.record_usage(db, &credential.uuid);
                    }
                    if request.stream {
                        build_anthropic_stream_response(&request.model, &parsed)
                    } else {
                        build_anthropic_response(&request.model, &parsed)
                    }
                }
                Err(e) => {
                    // 记录 API 调用失败
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::OpenAIKey { api_key, base_url } => {
            let openai = OpenAICustomProvider::with_config(api_key.clone(), base_url.clone());
            let openai_request = convert_anthropic_to_openai(request);
            match openai.call_api(&openai_request).await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.text().await {
                            Ok(body) => {
                                if let Ok(openai_resp) =
                                    serde_json::from_str::<serde_json::Value>(&body)
                                {
                                    let content = openai_resp["choices"][0]["message"]["content"]
                                        .as_str()
                                        .unwrap_or("");
                                    let parsed = CWParsedResponse {
                                        content: content.to_string(),
                                        tool_calls: Vec::new(),
                                        usage_credits: 0.0,
                                        context_usage_percentage: 0.0,
                                    };
                                    // 记录成功
                                    if let Some(db) = &state.db {
                                        let _ = state.pool_service.mark_healthy(
                                            db,
                                            &credential.uuid,
                                            Some(&request.model),
                                        );
                                        let _ =
                                            state.pool_service.record_usage(db, &credential.uuid);
                                    }
                                    if request.stream {
                                        build_anthropic_stream_response(&request.model, &parsed)
                                    } else {
                                        build_anthropic_response(&request.model, &parsed)
                                    }
                                } else {
                                    // 记录解析失败
                                    if let Some(db) = &state.db {
                                        let _ = state.pool_service.mark_unhealthy(
                                            db,
                                            &credential.uuid,
                                            Some("Failed to parse OpenAI response"),
                                        );
                                    }
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": "Failed to parse OpenAI response"}})),
                                    )
                                        .into_response()
                                }
                            }
                            Err(e) => {
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_unhealthy(
                                        db,
                                        &credential.uuid,
                                        Some(&e.to_string()),
                                    );
                                }
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                )
                                    .into_response()
                            }
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_unhealthy(
                                db,
                                &credential.uuid,
                                Some(&body),
                            );
                        }
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": body}})),
                        )
                            .into_response()
                    }
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::ClaudeKey { api_key, base_url } => {
            // 打印 Claude 代理 URL 用于调试
            let actual_base_url = base_url.as_deref().unwrap_or("https://api.anthropic.com");
            let claude = ClaudeCustomProvider::with_config(api_key.clone(), base_url.clone());
            let request_url = claude.get_base_url();
            state.logs.write().await.add(
                "info",
                &format!(
                    "[CLAUDE] 使用 Claude API 代理: base_url={} -> {}/v1/messages credential_uuid={}",
                    actual_base_url,
                    request_url,
                    &credential.uuid[..8]
                ),
            );
            // 打印请求参数
            let request_json = serde_json::to_string(request).unwrap_or_default();
            state.logs.write().await.add(
                "debug",
                &format!(
                    "[CLAUDE] 请求参数: {}",
                    &request_json.chars().take(500).collect::<String>()
                ),
            );
            match claude.call_api(request).await {
                Ok(resp) => {
                    let status = resp.status();
                    // 打印响应状态
                    state.logs.write().await.add(
                        "info",
                        &format!(
                            "[CLAUDE] 响应状态: status={} model={}",
                            status,
                            request.model
                        ),
                    );
                    match resp.text().await {
                        Ok(body) => {
                            if status.is_success() {
                                // 打印响应内容预览
                                state.logs.write().await.add(
                                    "debug",
                                    &format!(
                                        "[CLAUDE] 响应内容: {}",
                                        &body.chars().take(500).collect::<String>()
                                    ),
                                );
                                // 记录成功
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_healthy(
                                        db,
                                        &credential.uuid,
                                        Some(&request.model),
                                    );
                                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                                }
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header(header::CONTENT_TYPE, "application/json")
                                    .body(Body::from(body))
                                    .unwrap_or_else(|_| {
                                        (
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                            Json(serde_json::json!({"error": {"message": "Failed to build response"}})),
                                        )
                                            .into_response()
                                    })
                            } else {
                                state.logs.write().await.add(
                                    "error",
                                    &format!(
                                        "[CLAUDE] 请求失败: status={} body={}",
                                        status,
                                        &body.chars().take(200).collect::<String>()
                                    ),
                                );
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_unhealthy(
                                        db,
                                        &credential.uuid,
                                        Some(&body),
                                    );
                                }
                                (
                                    StatusCode::from_u16(status.as_u16())
                                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                                    Json(serde_json::json!({"error": {"message": body}})),
                                )
                                    .into_response()
                            }
                        }
                        Err(e) => {
                            state.logs.write().await.add(
                                "error",
                                &format!("[CLAUDE] 读取响应失败: {}", e),
                            );
                            if let Some(db) = &state.db {
                                let _ = state.pool_service.mark_unhealthy(
                                    db,
                                    &credential.uuid,
                                    Some(&e.to_string()),
                                );
                            }
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response()
                        }
                    }
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::VertexKey { api_key, base_url, .. } => {
            // Vertex AI uses Gemini-compatible API, convert Anthropic to OpenAI format first
            let openai_request = convert_anthropic_to_openai(request);
            let vertex = VertexProvider::with_config(api_key.clone(), base_url.clone());
            match vertex.chat_completions(&serde_json::to_value(&openai_request).unwrap_or_default()).await {
                Ok(resp) => {
                    let status = resp.status();
                    match resp.text().await {
                        Ok(body) => {
                            if status.is_success() {
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_healthy(db, &credential.uuid, Some(&request.model));
                                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                                }
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header(header::CONTENT_TYPE, "application/json")
                                    .body(Body::from(body))
                                    .unwrap_or_else(|_| {
                                        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": "Failed to build response"}}))).into_response()
                                    })
                            } else {
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&body));
                                }
                                (StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(serde_json::json!({"error": {"message": body}}))).into_response()
                            }
                        }
                        Err(e) => {
                            if let Some(db) = &state.db {
                                let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                            }
                            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response()
                        }
                    }
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                    }
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response()
                }
            }
        }
        // Gemini API Key credentials - not supported for Anthropic format
        CredentialData::GeminiApiKey { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "Gemini API Key credentials do not support Anthropic format"}})),
            )
                .into_response()
        }
        // 新增的凭证类型暂不支持 Anthropic 格式
        CredentialData::CodexOAuth { .. }
        | CredentialData::ClaudeOAuth { .. }
        | CredentialData::IFlowOAuth { .. }
        | CredentialData::IFlowCookie { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "This credential type does not support Anthropic format yet"}})),
            )
                .into_response()
        }
    }
}
/// 根据凭证调用 Provider (OpenAI 格式)
pub async fn call_provider_openai(
    state: &AppState,
    credential: &ProviderCredential,
    request: &ChatCompletionRequest,
) -> Response {
    let start_time = std::time::Instant::now();
    match &credential.credential {
        CredentialData::KiroOAuth { creds_file_path } => {
            let mut kiro = KiroProvider::new();
            if let Err(e) = kiro.load_credentials_from_path(creds_file_path).await {
                // 记录凭证加载失败
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                }
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": format!("Failed to load Kiro credentials: {}", e)}})),
                )
                    .into_response();
            }
            if let Err(e) = kiro.refresh_token().await {
                // 记录 Token 刷新失败
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&format!("Token refresh failed: {}", e)));
                }
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                )
                    .into_response();
            }
            match kiro.call_api(request).await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        // 记录成功
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_healthy(db, &credential.uuid, Some(&request.model));
                            let _ = state.pool_service.record_usage(db, &credential.uuid);
                        }
                        match resp.text().await {
                            Ok(body) => {
                                let parsed = parse_cw_response(&body);
                                let has_tool_calls = !parsed.tool_calls.is_empty();
                                let message = if has_tool_calls {
                                    serde_json::json!({
                                        "role": "assistant",
                                        "content": if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::json!(parsed.content) },
                                        "tool_calls": parsed.tool_calls.iter().map(|tc| {
                                            serde_json::json!({
                                                "id": tc.id,
                                                "type": "function",
                                                "function": {
                                                    "name": tc.function.name,
                                                    "arguments": tc.function.arguments
                                                }
                                            })
                                        }).collect::<Vec<_>>()
                                    })
                                } else {
                                    serde_json::json!({
                                        "role": "assistant",
                                        "content": parsed.content
                                    })
                                };
                                Json(serde_json::json!({
                                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                                    "object": "chat.completion",
                                    "created": std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_secs(),
                                    "model": request.model,
                                    "choices": [{
                                        "index": 0,
                                        "message": message,
                                        "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                                    }],
                                    "usage": {
                                        "prompt_tokens": 0,
                                        "completion_tokens": 0,
                                        "total_tokens": 0
                                    }
                                }))
                                .into_response()
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response(),
                        }
                    } else {
                        // 记录 API 调用失败
                        let body = resp.text().await.unwrap_or_default();
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&format!("HTTP {}: {}", status, safe_truncate(&body, 100))));
                        }
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": body}})),
                        )
                            .into_response()
                    }
                }
                Err(e) => {
                    // 记录请求错误
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::GeminiOAuth { .. } => {
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Gemini OAuth routing not yet implemented."}})),
            )
                .into_response()
        }
        CredentialData::QwenOAuth { .. } => {
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Qwen OAuth routing not yet implemented."}})),
            )
                .into_response()
        }
        CredentialData::AntigravityOAuth { creds_file_path, project_id } => {
            let mut antigravity = AntigravityProvider::new();
            if let Err(e) = antigravity.load_credentials_from_path(creds_file_path).await {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": format!("Failed to load Antigravity credentials: {}", e)}})),
                )
                    .into_response();
            }
            // 检查并刷新 token
            if antigravity.is_token_expiring_soon() {
                if let Err(e) = antigravity.refresh_token().await {
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                    )
                        .into_response();
                }
            }
            // 设置项目 ID
            if let Some(pid) = project_id {
                antigravity.project_id = Some(pid.clone());
            } else if let Err(e) = antigravity.discover_project().await {
                tracing::warn!("[Antigravity] Failed to discover project: {}", e);
            }
            // 获取 project_id 用于请求
            let proj_id = antigravity.project_id.clone().unwrap_or_default();
            // 转换请求格式
            let antigravity_request = convert_openai_to_antigravity_with_context(request, &proj_id);
            match antigravity.generate_content(&request.model, &antigravity_request).await {
                Ok(resp) => {
                    let openai_response = convert_antigravity_to_openai_response(&resp, &request.model);
                    Json(openai_response).into_response()
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                )
                    .into_response(),
            }
        }
        CredentialData::OpenAIKey { api_key, base_url } => {
            let openai = OpenAICustomProvider::with_config(api_key.clone(), base_url.clone());
            match openai.call_api(request).await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.text().await {
                            Ok(body) => {
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                                    Json(json).into_response()
                                } else {
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": "Invalid JSON response"}})),
                                    )
                                        .into_response()
                                }
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response(),
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": body}})),
                        )
                            .into_response()
                    }
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                )
                    .into_response(),
            }
        }
        CredentialData::ClaudeKey { api_key, base_url } => {
            // 打印 Claude 代理 URL 用于调试
            let actual_base_url = base_url.as_deref().unwrap_or("https://api.anthropic.com");
            tracing::info!(
                "[CLAUDE] 使用 Claude API 代理: base_url={} credential_uuid={}",
                actual_base_url,
                &credential.uuid[..8]
            );
            let claude = ClaudeCustomProvider::with_config(api_key.clone(), base_url.clone());
            match claude.call_openai_api(request).await {
                Ok(resp) => Json(resp).into_response(),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                )
                    .into_response(),
            }
        }
        CredentialData::VertexKey { api_key, base_url, model_aliases } => {
            // Resolve model alias if present
            let resolved_model = model_aliases.get(&request.model).cloned().unwrap_or_else(|| request.model.clone());
            let mut modified_request = request.clone();
            modified_request.model = resolved_model;
            let vertex = VertexProvider::with_config(api_key.clone(), base_url.clone());
            match vertex.chat_completions(&serde_json::to_value(&modified_request).unwrap_or_default()).await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.text().await {
                            Ok(body) => {
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                                    Json(json).into_response()
                                } else {
                                    (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": "Invalid JSON response"}}))).into_response()
                                }
                            }
                            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response(),
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": body}}))).into_response()
                    }
                }
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response(),
            }
        }
        // Gemini API Key credentials - not supported for OpenAI format yet
        CredentialData::GeminiApiKey { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "Gemini API Key credentials do not support OpenAI format yet"}})),
            )
                .into_response()
        }
        // 新增的凭证类型暂不支持 OpenAI 格式
        CredentialData::CodexOAuth { .. }
        | CredentialData::ClaudeOAuth { .. }
        | CredentialData::IFlowOAuth { .. }
        | CredentialData::IFlowCookie { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "This credential type does not support OpenAI format yet"}})),
            )
                .into_response()
        }
    }
}
