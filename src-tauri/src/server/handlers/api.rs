//! API 端点处理器
//!
//! 处理 OpenAI 和 Anthropic 格式的 API 请求

use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderMap, StatusCode},
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
use crate::processor::RequestContext;
use crate::providers::{AntigravityProvider, GeminiProvider, KiroProvider, QwenProvider};
use crate::server::{record_request_telemetry, record_token_usage, AppState};
use crate::server_utils::{
    build_anthropic_response, build_anthropic_stream_response, message_content_len,
    parse_cw_response, safe_truncate,
};
use crate::telemetry::RequestStatus;
use crate::ProviderType;

use super::{call_provider_anthropic, call_provider_openai};

/// OpenAI 格式的 API key 验证
pub async fn verify_api_key(
    headers: &HeaderMap,
    expected_key: &str,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    let auth = headers
        .get("authorization")
        .or_else(|| headers.get("x-api-key"))
        .and_then(|v| v.to_str().ok());

    let key = match auth {
        Some(s) if s.starts_with("Bearer ") => &s[7..],
        Some(s) => s,
        None => {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": {"message": "No API key provided"}})),
            ))
        }
    };

    if key != expected_key {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": {"message": "Invalid API key"}})),
        ));
    }

    Ok(())
}

/// Anthropic 格式的 API key 验证
pub async fn verify_api_key_anthropic(
    headers: &HeaderMap,
    expected_key: &str,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    let auth = headers
        .get("x-api-key")
        .or_else(|| headers.get("authorization"))
        .and_then(|v| v.to_str().ok());

    let key = match auth {
        Some(s) if s.starts_with("Bearer ") => &s[7..],
        Some(s) => s,
        None => {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "No API key provided. Please set the x-api-key header."
                    }
                })),
            ))
        }
    };

    if key != expected_key {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid API key"
                }
            })),
        ));
    }

    Ok(())
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(e) = verify_api_key(&headers, &state.api_key).await {
        state
            .logs
            .write()
            .await
            .add("warn", "Unauthorized request to /v1/chat/completions");
        return e.into_response();
    }

    // 创建请求上下文
    let mut ctx = RequestContext::new(request.model.clone()).with_stream(request.stream);

    state.logs.write().await.add(
        "info",
        &format!(
            "POST /v1/chat/completions request_id={} model={} stream={}",
            ctx.request_id, request.model, request.stream
        ),
    );

    // 使用 RequestProcessor 解析模型别名和路由
    let provider = state.processor.resolve_and_route(&mut ctx).await;

    // 更新请求中的模型名为解析后的模型
    if ctx.resolved_model != ctx.original_model {
        request.model = ctx.resolved_model.clone();
        state.logs.write().await.add(
            "info",
            &format!(
                "[MAPPER] request_id={} alias={} -> model={}",
                ctx.request_id, ctx.original_model, ctx.resolved_model
            ),
        );
    }

    // 应用参数注入
    let injection_enabled = *state.injection_enabled.read().await;
    if injection_enabled {
        let injector = state.processor.injector.read().await;
        let mut payload = serde_json::to_value(&request).unwrap_or_default();
        let result = injector.inject(&request.model, &mut payload);
        if result.has_injections() {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[INJECT] request_id={} applied_rules={:?} injected_params={:?}",
                    ctx.request_id, result.applied_rules, result.injected_params
                ),
            );
            // 更新请求
            if let Ok(updated) = serde_json::from_value(payload) {
                request = updated;
            }
        }
    }

    // 获取当前默认 provider（用于凭证池选择）
    let default_provider = state.default_provider.read().await.clone();

    // 记录路由结果
    state.logs.write().await.add(
        "info",
        &format!(
            "[ROUTE] request_id={} model={} provider={}",
            ctx.request_id, ctx.resolved_model, provider
        ),
    );

    // 尝试从凭证池中选择凭证
    let credential = match &state.db {
        Some(db) => state
            .pool_service
            .select_credential(db, &default_provider, Some(&request.model))
            .ok()
            .flatten(),
        None => None,
    };

    // 如果找到凭证池中的凭证，使用它
    if let Some(cred) = credential {
        state.logs.write().await.add(
            "info",
            &format!(
                "[ROUTE] Using pool credential: type={} name={:?} uuid={}",
                cred.provider_type,
                cred.name,
                &cred.uuid[..8]
            ),
        );
        let response = call_provider_openai(&state, &cred, &request).await;

        // 记录请求统计
        let is_success = response.status().is_success();
        let status = if is_success {
            crate::telemetry::RequestStatus::Success
        } else {
            crate::telemetry::RequestStatus::Failed
        };
        record_request_telemetry(&state, &ctx, status, None);

        // 如果成功，记录估算的 Token 使用量
        if is_success {
            let estimated_input_tokens = request
                .messages
                .iter()
                .map(|m| {
                    let content_len = match &m.content {
                        Some(c) => message_content_len(c),
                        None => 0,
                    };
                    content_len / 4
                })
                .sum::<usize>() as u32;
            // 输出 Token 使用估算值（假设平均响应长度）
            let estimated_output_tokens = 100u32;
            record_token_usage(
                &state,
                &ctx,
                Some(estimated_input_tokens),
                Some(estimated_output_tokens),
            );
        }

        return response;
    }

    // 回退到旧的单凭证模式
    state.logs.write().await.add(
        "debug",
        &format!(
            "[ROUTE] No pool credential found for '{}', using legacy mode",
            default_provider
        ),
    );

    // 检查是否需要刷新 token（无 token 或即将过期）
    {
        let _guard = state.kiro_refresh_lock.lock().await;
        let mut kiro = state.kiro.write().await;
        let needs_refresh =
            kiro.credentials.access_token.is_none() || kiro.is_token_expiring_soon();
        if needs_refresh {
            if let Err(e) = kiro.refresh_token().await {
                state
                    .logs
                    .write()
                    .await
                    .add("error", &format!("Token refresh failed: {e}"));
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                ).into_response();
            }
        }
    }

    let kiro = state.kiro.read().await;

    match kiro.call_api(&request).await {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                match resp.text().await {
                    Ok(body) => {
                        let parsed = parse_cw_response(&body);
                        let has_tool_calls = !parsed.tool_calls.is_empty();

                        state.logs.write().await.add(
                            "info",
                            &format!(
                                "Request completed: content_len={}, tool_calls={}",
                                parsed.content.len(),
                                parsed.tool_calls.len()
                            ),
                        );

                        // 构建消息
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

                        // 估算 Token 数量（基于字符数，约 4 字符 = 1 token）
                        let estimated_output_tokens = (parsed.content.len() / 4) as u32;
                        // 估算输入 Token（基于请求消息）
                        let estimated_input_tokens = request
                            .messages
                            .iter()
                            .map(|m| {
                                let content_len = match &m.content {
                                    Some(c) => message_content_len(c),
                                    None => 0,
                                };
                                content_len / 4
                            })
                            .sum::<usize>()
                            as u32;

                        let response = serde_json::json!({
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
                                "prompt_tokens": estimated_input_tokens,
                                "completion_tokens": estimated_output_tokens,
                                "total_tokens": estimated_input_tokens + estimated_output_tokens
                            }
                        });
                        // 记录成功请求统计
                        record_request_telemetry(
                            &state,
                            &ctx,
                            crate::telemetry::RequestStatus::Success,
                            None,
                        );
                        // 记录 Token 使用量
                        record_token_usage(
                            &state,
                            &ctx,
                            Some(estimated_input_tokens),
                            Some(estimated_output_tokens),
                        );
                        Json(response).into_response()
                    }
                    Err(e) => {
                        // 记录失败请求统计
                        record_request_telemetry(
                            &state,
                            &ctx,
                            crate::telemetry::RequestStatus::Failed,
                            Some(e.to_string()),
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else if status.as_u16() == 403 || status.as_u16() == 402 {
                // Token 过期或账户问题，尝试重新加载凭证并刷新
                drop(kiro);
                let _guard = state.kiro_refresh_lock.lock().await;
                let mut kiro = state.kiro.write().await;
                state.logs.write().await.add(
                    "warn",
                    &format!(
                        "[AUTH] Got {}, reloading credentials and attempting token refresh...",
                        status.as_u16()
                    ),
                );

                // 先重新加载凭证文件（可能用户换了账户）
                if let Err(e) = kiro.load_credentials().await {
                    state.logs.write().await.add(
                        "error",
                        &format!("[AUTH] Failed to reload credentials: {e}"),
                    );
                }

                match kiro.refresh_token().await {
                    Ok(_) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("info", "[AUTH] Token refreshed successfully after reload");
                        // 重试请求
                        drop(kiro);
                        let kiro = state.kiro.read().await;
                        match kiro.call_api(&request).await {
                            Ok(retry_resp) => {
                                if retry_resp.status().is_success() {
                                    match retry_resp.text().await {
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

                                            let response = serde_json::json!({
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
                                            });
                                            return Json(response).into_response();
                                        }
                                        Err(e) => return (
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                        ).into_response(),
                                    }
                                }
                                let body = retry_resp.text().await.unwrap_or_default();
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": format!("Retry failed: {}", body)}})),
                                ).into_response()
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response(),
                        }
                    }
                    Err(e) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("error", &format!("[AUTH] Token refresh failed: {e}"));
                        (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                        )
                            .into_response()
                    }
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                state.logs.write().await.add(
                    "error",
                    &format!("Upstream error {}: {}", status, safe_truncate(&body, 200)),
                );
                (
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(serde_json::json!({"error": {"message": format!("Upstream error: {}", body)}}))
                ).into_response()
            }
        }
        Err(e) => {
            state
                .logs
                .write()
                .await
                .add("error", &format!("API call failed: {e}"));
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string()}})),
            )
                .into_response()
        }
    }
}

pub async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut request): Json<AnthropicMessagesRequest>,
) -> Response {
    // 使用 Anthropic 格式的认证验证（优先检查 x-api-key）
    if let Err(e) = verify_api_key_anthropic(&headers, &state.api_key).await {
        state
            .logs
            .write()
            .await
            .add("warn", "Unauthorized request to /v1/messages");
        return e.into_response();
    }

    // 创建请求上下文
    let mut ctx = RequestContext::new(request.model.clone()).with_stream(request.stream);

    // 详细记录请求信息
    let msg_count = request.messages.len();
    let has_tools = request.tools.as_ref().map(|t| t.len()).unwrap_or(0);
    let has_system = request.system.is_some();
    state.logs.write().await.add(
        "info",
        &format!(
            "[REQ] POST /v1/messages request_id={} model={} stream={} messages={} tools={} has_system={}",
            ctx.request_id, request.model, request.stream, msg_count, has_tools, has_system
        ),
    );

    // 使用 RequestProcessor 解析模型别名和路由
    let provider = state.processor.resolve_and_route(&mut ctx).await;

    // 更新请求中的模型名为解析后的模型
    if ctx.resolved_model != ctx.original_model {
        request.model = ctx.resolved_model.clone();
        state.logs.write().await.add(
            "info",
            &format!(
                "[MAPPER] request_id={} alias={} -> model={}",
                ctx.request_id, ctx.original_model, ctx.resolved_model
            ),
        );
    }

    // 记录最后一条消息的角色和内容预览
    if let Some(last_msg) = request.messages.last() {
        let content_preview = match &last_msg.content {
            serde_json::Value::String(s) => s.chars().take(100).collect::<String>(),
            serde_json::Value::Array(arr) => {
                if let Some(first) = arr.first() {
                    if let Some(text) = first.get("text").and_then(|t| t.as_str()) {
                        text.chars().take(100).collect::<String>()
                    } else {
                        format!("[{} blocks]", arr.len())
                    }
                } else {
                    "[empty]".to_string()
                }
            }
            _ => "[unknown]".to_string(),
        };
        state.logs.write().await.add(
            "debug",
            &format!(
                "[REQ] request_id={} last_message: role={} content={}",
                ctx.request_id, last_msg.role, content_preview
            ),
        );
    }

    // 应用参数注入
    let injection_enabled = *state.injection_enabled.read().await;
    if injection_enabled {
        let injector = state.processor.injector.read().await;
        let mut payload = serde_json::to_value(&request).unwrap_or_default();
        let result = injector.inject(&request.model, &mut payload);
        if result.has_injections() {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[INJECT] request_id={} applied_rules={:?} injected_params={:?}",
                    ctx.request_id, result.applied_rules, result.injected_params
                ),
            );
            // 更新请求
            if let Ok(updated) = serde_json::from_value(payload) {
                request = updated;
            }
        }
    }

    // 获取当前默认 provider（用于凭证池选择）
    let default_provider = state.default_provider.read().await.clone();

    // 记录路由结果
    state.logs.write().await.add(
        "info",
        &format!(
            "[ROUTE] request_id={} model={} provider={}",
            ctx.request_id, ctx.resolved_model, provider
        ),
    );

    // 尝试从凭证池中选择凭证
    let credential = match &state.db {
        Some(db) => {
            // 根据 default_provider 配置选择凭证
            state
                .pool_service
                .select_credential(db, &default_provider, Some(&request.model))
                .ok()
                .flatten()
        }
        None => None,
    };

    // 如果找到凭证池中的凭证，使用它
    if let Some(cred) = credential {
        state.logs.write().await.add(
            "info",
            &format!(
                "[ROUTE] Using pool credential: type={} name={:?} uuid={}",
                cred.provider_type,
                cred.name,
                &cred.uuid[..8]
            ),
        );
        let response = call_provider_anthropic(&state, &cred, &request).await;

        // 记录请求统计
        let is_success = response.status().is_success();
        let status = if is_success {
            crate::telemetry::RequestStatus::Success
        } else {
            crate::telemetry::RequestStatus::Failed
        };
        record_request_telemetry(&state, &ctx, status, None);

        // 如果成功，记录估算的 Token 使用量
        if is_success {
            let estimated_input_tokens = request
                .messages
                .iter()
                .map(|m| {
                    let content_len = match &m.content {
                        serde_json::Value::String(s) => s.len(),
                        serde_json::Value::Array(arr) => arr
                            .iter()
                            .filter_map(|v| v.get("text").and_then(|t| t.as_str()))
                            .map(|s| s.len())
                            .sum(),
                        _ => 0,
                    };
                    content_len / 4
                })
                .sum::<usize>() as u32;
            // 输出 Token 使用估算值
            let estimated_output_tokens = 100u32;
            record_token_usage(
                &state,
                &ctx,
                Some(estimated_input_tokens),
                Some(estimated_output_tokens),
            );
        }

        return response;
    }

    // 回退到旧的单凭证模式
    state.logs.write().await.add(
        "debug",
        &format!(
            "[ROUTE] No pool credential found for '{}', using legacy mode",
            default_provider
        ),
    );

    // 检查是否需要刷新 token（无 token 或即将过期）
    {
        let _guard = state.kiro_refresh_lock.lock().await;
        let mut kiro = state.kiro.write().await;
        let needs_refresh =
            kiro.credentials.access_token.is_none() || kiro.is_token_expiring_soon();
        if needs_refresh {
            state.logs.write().await.add(
                "info",
                "[AUTH] No access token or token expiring soon, attempting refresh...",
            );
            if let Err(e) = kiro.refresh_token().await {
                state
                    .logs
                    .write()
                    .await
                    .add("error", &format!("[AUTH] Token refresh failed: {e}"));
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                )
                    .into_response();
            }
            state
                .logs
                .write()
                .await
                .add("info", "[AUTH] Token refreshed successfully");
        }
    }

    // 转换为 OpenAI 格式
    let openai_request = convert_anthropic_to_openai(&request);

    // 记录转换后的请求信息
    state.logs.write().await.add(
        "debug",
        &format!(
            "[CONVERT] OpenAI format: messages={} tools={} stream={}",
            openai_request.messages.len(),
            openai_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            openai_request.stream
        ),
    );

    let kiro = state.kiro.read().await;

    match kiro.call_api(&openai_request).await {
        Ok(resp) => {
            let status = resp.status();
            state
                .logs
                .write()
                .await
                .add("info", &format!("[RESP] Upstream status: {status}"));

            if status.is_success() {
                match resp.bytes().await {
                    Ok(bytes) => {
                        // 使用 lossy 转换，避免无效 UTF-8 导致崩溃
                        let body = String::from_utf8_lossy(&bytes).to_string();

                        // 记录原始响应长度
                        state.logs.write().await.add(
                            "debug",
                            &format!("[RESP] Raw body length: {} bytes", bytes.len()),
                        );

                        // 保存原始响应到文件用于调试
                        let request_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
                        state.logs.read().await.log_raw_response(&request_id, &body);
                        state.logs.write().await.add(
                            "debug",
                            &format!("[RESP] Raw response saved to raw_response_{request_id}.txt"),
                        );

                        // 记录响应的前200字符用于调试（减少日志量）
                        let preview: String =
                            body.chars().filter(|c| !c.is_control()).take(200).collect();
                        state
                            .logs
                            .write()
                            .await
                            .add("debug", &format!("[RESP] Body preview: {preview}"));

                        let parsed = parse_cw_response(&body);

                        // 详细记录解析结果
                        state.logs.write().await.add(
                            "info",
                            &format!(
                                "[RESP] Parsed: content_len={}, tool_calls={}, content_preview={}",
                                parsed.content.len(),
                                parsed.tool_calls.len(),
                                parsed.content.chars().take(100).collect::<String>()
                            ),
                        );

                        // 记录 tool calls 详情
                        for (i, tc) in parsed.tool_calls.iter().enumerate() {
                            state.logs.write().await.add(
                                "debug",
                                &format!(
                                    "[RESP] Tool call {}: name={} id={}",
                                    i, tc.function.name, tc.id
                                ),
                            );
                        }

                        // 如果请求流式响应，返回 SSE 格式
                        if request.stream {
                            return build_anthropic_stream_response(&request.model, &parsed);
                        }

                        // 非流式响应
                        build_anthropic_response(&request.model, &parsed)
                    }
                    Err(e) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("error", &format!("[ERROR] Response body read failed: {e}"));
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else if status.as_u16() == 403 || status.as_u16() == 402 {
                // Token 过期或账户问题，尝试重新加载凭证并刷新
                drop(kiro);
                let _guard = state.kiro_refresh_lock.lock().await;
                let mut kiro = state.kiro.write().await;
                state.logs.write().await.add(
                    "warn",
                    &format!(
                        "[AUTH] Got {}, reloading credentials and attempting token refresh...",
                        status.as_u16()
                    ),
                );

                // 先重新加载凭证文件（可能用户换了账户）
                if let Err(e) = kiro.load_credentials().await {
                    state.logs.write().await.add(
                        "error",
                        &format!("[AUTH] Failed to reload credentials: {e}"),
                    );
                }

                match kiro.refresh_token().await {
                    Ok(_) => {
                        state.logs.write().await.add(
                            "info",
                            "[AUTH] Token refreshed successfully, retrying request...",
                        );
                        drop(kiro);
                        let kiro = state.kiro.read().await;
                        match kiro.call_api(&openai_request).await {
                            Ok(retry_resp) => {
                                let retry_status = retry_resp.status();
                                state.logs.write().await.add(
                                    "info",
                                    &format!("[RETRY] Response status: {retry_status}"),
                                );
                                if retry_resp.status().is_success() {
                                    match retry_resp.bytes().await {
                                        Ok(bytes) => {
                                            let body = String::from_utf8_lossy(&bytes).to_string();
                                            let parsed = parse_cw_response(&body);
                                            state.logs.write().await.add(
                                                "info",
                                                &format!(
                                                "[RETRY] Success: content_len={}, tool_calls={}",
                                                parsed.content.len(), parsed.tool_calls.len()
                                            ),
                                            );
                                            if request.stream {
                                                return build_anthropic_stream_response(
                                                    &request.model,
                                                    &parsed,
                                                );
                                            }
                                            return build_anthropic_response(
                                                &request.model,
                                                &parsed,
                                            );
                                        }
                                        Err(e) => {
                                            state.logs.write().await.add(
                                                "error",
                                                &format!("[RETRY] Body read failed: {e}"),
                                            );
                                            return (
                                                StatusCode::INTERNAL_SERVER_ERROR,
                                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                            )
                                                .into_response();
                                        }
                                    }
                                }
                                let body = retry_resp
                                    .bytes()
                                    .await
                                    .map(|b| String::from_utf8_lossy(&b).to_string())
                                    .unwrap_or_default();
                                state.logs.write().await.add(
                                    "error",
                                    &format!(
                                        "[RETRY] Failed with status {retry_status}: {}",
                                        safe_truncate(&body, 500)
                                    ),
                                );
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": format!("Retry failed: {}", body)}})),
                                )
                                    .into_response()
                            }
                            Err(e) => {
                                state
                                    .logs
                                    .write()
                                    .await
                                    .add("error", &format!("[RETRY] Request failed: {e}"));
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                )
                                    .into_response()
                            }
                        }
                    }
                    Err(e) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("error", &format!("[AUTH] Token refresh failed: {e}"));
                        (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                        )
                            .into_response()
                    }
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                state.logs.write().await.add(
                    "error",
                    &format!(
                        "[ERROR] Upstream error HTTP {}: {}",
                        status,
                        safe_truncate(&body, 500)
                    ),
                );
                (
                    StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(
                        serde_json::json!({"error": {"message": format!("Upstream error: {}", body)}}),
                    ),
                )
                    .into_response()
            }
        }
        Err(e) => {
            // 详细记录网络/连接错误
            let error_details = format!("{e:?}");
            state
                .logs
                .write()
                .await
                .add("error", &format!("[ERROR] Kiro API call failed: {e}"));
            state.logs.write().await.add(
                "debug",
                &format!("[ERROR] Full error details: {error_details}"),
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string()}})),
            )
                .into_response()
        }
    }
}
