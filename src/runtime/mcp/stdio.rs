use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;

use crate::runtime::jsonrpc;

use super::client::{
    CallToolResult, InitializeResult, McpClient, McpError, ServerCapabilities, ServerInfo,
    ToolDefinition,
};

// ---------------------------------------------------------------------------
// Protocol version
// ---------------------------------------------------------------------------

const PROTOCOL_VERSION: &str = "2025-11-25";

// ---------------------------------------------------------------------------
// StdioMcpClient — spawns a child process, speaks JSON-RPC over stdin/stdout
// ---------------------------------------------------------------------------

pub struct StdioMcpClient {
    io: Mutex<StdioIo>,
    tools: Vec<ToolDefinition>,
    next_id: AtomicU64,
    child: Mutex<Child>,
    capabilities: ServerCapabilities,
    server_info: ServerInfo,
    instructions: Option<String>,
}

struct StdioIo {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl StdioMcpClient {
    #[tracing::instrument(skip(args), fields(%command))]
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
            capabilities: ServerCapabilities::default(),
            server_info: ServerInfo::default(),
            instructions: None,
        };

        // 1. Send initialize request
        let init_params = serde_json::json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "substructure", "version": "0.1.0" }
        });
        let init_result = client.request("initialize", Some(init_params)).await?;

        // Parse the initialize response
        let init: InitializeResult = serde_json::from_value(init_result)
            .map_err(|e| McpError::Protocol(format!("failed to parse initialize result: {e}")))?;

        // Validate protocol version
        if init.protocol_version != PROTOCOL_VERSION {
            tracing::warn!(negotiated = %init.protocol_version, expected = %PROTOCOL_VERSION, "MCP protocol version mismatch");
        }

        client.capabilities = init.capabilities;
        client.server_info = init.server_info;
        client.instructions = init.instructions;

        // 2. Send initialized notification
        client.notify("notifications/initialized", None).await?;

        // 3. Fetch tool list (with pagination)
        client.tools = client.fetch_all_tools().await?;

        Ok(client)
    }

    /// Fetch all tools, following pagination cursors.
    async fn fetch_all_tools(&self) -> Result<Vec<ToolDefinition>, McpError> {
        let mut all_tools = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            let params = cursor.as_ref().map(|c| serde_json::json!({ "cursor": c }));
            let result = self.request("tools/list", params).await?;

            let tools: Vec<ToolDefinition> = serde_json::from_value(
                result
                    .get("tools")
                    .cloned()
                    .unwrap_or(serde_json::Value::Array(vec![])),
            )
            .map_err(|e| McpError::Protocol(format!("failed to parse tools/list: {e}")))?;
            all_tools.extend(tools);

            // Check for pagination
            match result.get("nextCursor").and_then(|v| v.as_str()) {
                Some(next) => cursor = Some(next.to_string()),
                None => break,
            }
        }

        Ok(all_tools)
    }

    async fn request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        let id_num = self.next_id.fetch_add(1, Ordering::Relaxed);
        let id = jsonrpc::Id::Number(id_num);
        let req = jsonrpc::Request::new(id.clone(), method, params);

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

        // Read response lines, skipping notifications and non-matching responses
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

            // Try to parse as a response
            if let Ok(resp) = serde_json::from_str::<jsonrpc::Response>(&buf) {
                if !resp.id_matches(&id) {
                    // Response for a different request — skip it
                    continue;
                }

                if let Some(err) = resp.error {
                    return Err(McpError::JsonRpc {
                        code: err.code,
                        message: err.message,
                        data: err.data,
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
        let notif = jsonrpc::Notification::new(method, params);
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

    fn capabilities(&self) -> &ServerCapabilities {
        &self.capabilities
    }

    fn server_info(&self) -> &ServerInfo {
        &self.server_info
    }

    fn instructions(&self) -> Option<&str> {
        self.instructions.as_deref()
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

    async fn refresh_tools(&mut self) -> Result<&[ToolDefinition], McpError> {
        self.tools = self.fetch_all_tools().await?;
        Ok(&self.tools)
    }

    async fn shutdown(&mut self) -> Result<(), McpError> {
        // Close stdin to signal the child process to exit
        let mut io = self.io.lock().await;
        let _ = io.stdin.shutdown().await;
        drop(io);

        let mut child = self.child.lock().await;

        // Wait briefly for the process to exit gracefully
        if tokio::time::timeout(Duration::from_secs(2), child.wait())
            .await
            .is_ok()
        {
            return Ok(());
        }

        // Send SIGTERM on Unix
        #[cfg(unix)]
        {
            if let Some(pid) = child.id() {
                let _ = std::process::Command::new("kill")
                    .args(["-TERM", &pid.to_string()])
                    .status();
            }
            if tokio::time::timeout(Duration::from_secs(3), child.wait())
                .await
                .is_ok()
            {
                return Ok(());
            }
        }

        // Force kill as last resort
        let _ = child.kill().await;
        let _ = child.wait().await;
        Ok(())
    }
}
