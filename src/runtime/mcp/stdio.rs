use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;

use super::client::{CallToolResult, McpClient, McpError, ToolDefinition};

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 types (internal)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct JsonRpcNotification {
    jsonrpc: String,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    #[allow(dead_code)]
    id: u64,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

// ---------------------------------------------------------------------------
// StdioMcpClient — spawns a child process, speaks JSON-RPC over stdin/stdout
// ---------------------------------------------------------------------------

pub struct StdioMcpClient {
    io: Mutex<StdioIo>,
    tools: Vec<ToolDefinition>,
    next_id: AtomicU64,
    #[allow(dead_code)]
    child: Mutex<Child>,
}

struct StdioIo {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl StdioMcpClient {
    pub async fn new(command: &str, args: &[String]) -> Result<Self, McpError> {
        let mut child = tokio::process::Command::new(command)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| McpError::Transport(format!("failed to spawn {command}: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("no stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("no stdout".into()))?;

        let mut client = StdioMcpClient {
            io: Mutex::new(StdioIo {
                stdin,
                stdout: BufReader::new(stdout),
            }),
            tools: Vec::new(),
            next_id: AtomicU64::new(1),
            child: Mutex::new(child),
        };

        // 1. Send initialize request
        let init_params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "substructure", "version": "0.1.0" }
        });
        let _init_result = client.request("initialize", Some(init_params)).await?;

        // 2. Send initialized notification
        client.notify("notifications/initialized", None).await?;

        // 3. Fetch tool list
        let tools_result = client.request("tools/list", None).await?;
        let tools: Vec<ToolDefinition> = serde_json::from_value(
            tools_result
                .get("tools")
                .cloned()
                .unwrap_or(serde_json::Value::Array(vec![])),
        )
        .map_err(|e| McpError::Protocol(format!("failed to parse tools/list: {e}")))?;
        client.tools = tools;

        Ok(client)
    }

    async fn request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id,
            method: method.into(),
            params,
        };

        let mut io = self.io.lock().await;
        let line = serde_json::to_string(&req)
            .map_err(|e| McpError::Transport(format!("serialize: {e}")))?;
        io.stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| McpError::Transport(format!("write: {e}")))?;
        io.stdin
            .write_all(b"\n")
            .await
            .map_err(|e| McpError::Transport(format!("write newline: {e}")))?;
        io.stdin
            .flush()
            .await
            .map_err(|e| McpError::Transport(format!("flush: {e}")))?;

        // Read response lines, skipping notifications (lines without "id")
        loop {
            let mut buf = String::new();
            let n = io
                .stdout
                .read_line(&mut buf)
                .await
                .map_err(|e| McpError::Transport(format!("read: {e}")))?;
            if n == 0 {
                return Err(McpError::Transport("unexpected EOF".into()));
            }

            // Try to parse as a response with our id
            if let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(&buf) {
                if let Some(err) = resp.error {
                    return Err(McpError::JsonRpc {
                        code: err.code,
                        message: err.message,
                    });
                }
                return Ok(resp.result.unwrap_or(serde_json::Value::Null));
            }
            // Otherwise it's a notification or something else — skip it
        }
    }

    async fn notify(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), McpError> {
        let notif = JsonRpcNotification {
            jsonrpc: "2.0".into(),
            method: method.into(),
            params,
        };
        let mut io = self.io.lock().await;
        let line = serde_json::to_string(&notif)
            .map_err(|e| McpError::Transport(format!("serialize: {e}")))?;
        io.stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| McpError::Transport(format!("write: {e}")))?;
        io.stdin
            .write_all(b"\n")
            .await
            .map_err(|e| McpError::Transport(format!("write newline: {e}")))?;
        io.stdin
            .flush()
            .await
            .map_err(|e| McpError::Transport(format!("flush: {e}")))?;
        Ok(())
    }
}

#[async_trait]
impl McpClient for StdioMcpClient {
    fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });
        let result = self.request("tools/call", Some(params)).await?;
        serde_json::from_value(result)
            .map_err(|e| McpError::Protocol(format!("failed to parse tools/call result: {e}")))
    }
}
