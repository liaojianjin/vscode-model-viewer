'use strict';

const vscode = require('vscode');
const http = require('node:http');
const net = require('node:net');
const path = require('node:path');
const { spawn } = require('node:child_process');

const activeSessions = new Set();

function activate(context) {
  const output = vscode.window.createOutputChannel('Remote Model Viewer');

  context.subscriptions.push(
    output,
    vscode.commands.registerCommand('remoteModelViewer.openModel', async (uri) => {
      await openModelPanel(context, output, uri);
    }),
    vscode.commands.registerCommand('remoteModelViewer.openActiveModel', async () => {
      const activeUri = vscode.window.activeTextEditor?.document?.uri;
      await openModelPanel(context, output, activeUri);
    }),
    {
      dispose: () => {
        for (const session of Array.from(activeSessions)) {
          disposeSession(session);
        }
      }
    }
  );
}

async function openModelPanel(context, output, targetUri) {
  const uri = targetUri ?? vscode.window.activeTextEditor?.document?.uri;
  if (!uri) {
    vscode.window.showErrorMessage('No file selected to open in Remote Model Viewer.');
    return;
  }

  try {
    const stat = await vscode.workspace.fs.stat(uri);
    if ((stat.type & vscode.FileType.File) === 0) {
      vscode.window.showErrorMessage('Remote Model Viewer can only open files.');
      return;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    vscode.window.showErrorMessage(`Unable to access the selected file: ${message}`);
    return;
  }

  const filePath = toHostFilePath(uri);
  const fileName = path.basename(filePath);
  const panel = vscode.window.createWebviewPanel(
    'remoteModelViewer.preview',
    `Netron: ${fileName}`,
    vscode.ViewColumn.Active,
    {
      enableScripts: true,
      retainContextWhenHidden: true
    }
  );

  panel.webview.html = getLoadingHtml(panel.webview, fileName);

  let panelDisposed = false;
  let session;
  panel.onDidDispose(() => {
    panelDisposed = true;
    if (session && activeSessions.delete(session)) {
      disposeSession(session);
    }
  });

  try {
    session = await startSession(context, output, filePath);
    if (panelDisposed) {
      disposeSession(session);
      return;
    }
    activeSessions.add(session);
    panel.webview.html = getViewerHtml(panel.webview, {
      fileName,
      externalUri: session.externalUri
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    output.appendLine(`[error] ${message}`);
    panel.webview.html = getErrorHtml(panel.webview, fileName, message);
    vscode.window.showErrorMessage(`Unable to open ${fileName} in Remote Model Viewer. See "Remote Model Viewer" output for details.`);
  }
}

async function startSession(context, output, filePath) {
  const config = vscode.workspace.getConfiguration('remoteModelViewer');
  const timeoutMs = Math.max(1000, Number(config.get('startupTimeoutMs', 15000)) || 15000);
  const verbose = Boolean(config.get('enableVerboseServerLog', false));
  const configuredPython = String(config.get('pythonCommand', 'python3')).trim();
  const pythonCandidates = Array.from(new Set([configuredPython, 'python3', 'python'].filter(Boolean)));
  const port = await getFreePort();
  const scriptPath = path.join(context.extensionPath, 'scripts', 'netron_server.py');
  const attempts = [];

  for (const pythonCommand of pythonCandidates) {
    const args = [
      scriptPath,
      '--file',
      filePath,
      '--host',
      '127.0.0.1',
      '--port',
      String(port)
    ];
    if (verbose) {
      args.push('--log');
    }

    output.appendLine(`[info] Starting Netron via "${pythonCommand}" for ${filePath} on 127.0.0.1:${port}`);

    try {
      const child = await spawnAndWaitForServer({
        command: pythonCommand,
        args,
        cwd: path.dirname(filePath),
        timeoutMs,
        port,
        output
      });
      const externalUri = await vscode.env.asExternalUri(vscode.Uri.parse(`http://127.0.0.1:${port}/`));
      output.appendLine(`[info] Netron ready at ${externalUri.toString(true)}`);
      return {
        child,
        port,
        externalUri
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      attempts.push(`${pythonCommand}: ${message}`);
      output.appendLine(`[warn] ${pythonCommand} failed: ${message}`);
    }
  }

  throw new Error(
    [
      'Failed to start Netron on the extension host.',
      'Ensure the Python package "netron" is installed on the local or remote machine running the extension (for example: pip install netron).',
      'Attempt details:',
      ...attempts.map((attempt) => `- ${attempt}`)
    ].join('\n')
  );
}

function spawnAndWaitForServer({ command, args, cwd, timeoutMs, port, output }) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: process.env,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let settled = false;
    let lastStdout = '';
    let lastStderr = '';

    child.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      lastStdout = trimBuffer(lastStdout + text);
      for (const line of text.split(/\r?\n/)) {
        if (line) {
          output.appendLine(`[netron stdout] ${line}`);
        }
      }
    });

    child.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      lastStderr = trimBuffer(lastStderr + text);
      for (const line of text.split(/\r?\n/)) {
        if (line) {
          output.appendLine(`[netron stderr] ${line}`);
        }
      }
    });

    child.once('error', (error) => {
      if (settled) {
        return;
      }
      settled = true;
      reject(error);
    });

    child.once('exit', (code, signal) => {
      if (settled) {
        return;
      }
      settled = true;
      reject(
        new Error(
          [
            `Netron process exited before the server became ready (code=${String(code)}, signal=${String(signal)}).`,
            lastStderr || lastStdout || 'The process exited without additional output.'
          ].join(' ')
        )
      );
    });

    const startTime = Date.now();
    const tick = async () => {
      if (settled) {
        return;
      }
      if (Date.now() - startTime > timeoutMs) {
        settled = true;
        child.kill();
        reject(
          new Error(
            [
              `Timed out after ${timeoutMs}ms while waiting for Netron on port ${port}.`,
              lastStderr || lastStdout || 'No additional output was captured.'
            ].join(' ')
          )
        );
        return;
      }

      try {
        await probeHttpServer(port);
        settled = true;
        resolve(child);
      } catch (_error) {
        setTimeout(() => {
          void tick();
        }, 250);
      }
    };

    setTimeout(() => {
      void tick();
    }, 150);
  });
}

function probeHttpServer(port) {
  return new Promise((resolve, reject) => {
    const request = http.get(
      {
        host: '127.0.0.1',
        port,
        path: '/'
      },
      (response) => {
        response.resume();
        resolve();
      }
    );

    request.setTimeout(800, () => {
      request.destroy(new Error('Connection timed out.'));
    });
    request.on('error', reject);
  });
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (!address || typeof address === 'string') {
        server.close(() => reject(new Error('Failed to allocate a free TCP port.')));
        return;
      }
      const { port } = address;
      server.close((error) => {
        if (error) {
          reject(error);
          return;
        }
        resolve(port);
      });
    });
  });
}

function toHostFilePath(uri) {
  if (uri.scheme === 'file') {
    return uri.fsPath;
  }
  if (uri.scheme === 'vscode-remote') {
    return uri.path;
  }
  return uri.fsPath || uri.path;
}

function disposeSession(session) {
  try {
    session.child.stdin.end();
  } catch (_error) {
    // Ignore shutdown errors while tearing down the helper process.
  }

  if (!session.child.killed) {
    session.child.kill();
  }
}

function trimBuffer(value) {
  const text = value.trim();
  return text.length > 1200 ? text.slice(text.length - 1200) : text;
}

function getLoadingHtml(webview, fileName) {
  return getShellHtml(webview, {
    fileName,
    body: `
      <div class="state-card">
        <div class="state-title">Launching Netron</div>
        <div class="state-copy">${escapeHtml(fileName)}</div>
      </div>
    `,
    script: ''
  });
}

function getErrorHtml(webview, fileName, message) {
  return getShellHtml(webview, {
    fileName,
    body: `
      <div class="state-card error">
        <div class="state-title">Unable to start Netron</div>
        <pre class="state-copy">${escapeHtml(message)}</pre>
      </div>
    `,
    script: ''
  });
}

function getViewerHtml(webview, { fileName, externalUri }) {
  const nonce = createNonce();
  const iframeSource = externalUri.toString();
  const frameOrigin = new URL(iframeSource).origin;

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta
    http-equiv="Content-Security-Policy"
    content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; frame-src ${frameOrigin};"
  />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${escapeHtml(fileName)}</title>
  <style>
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: var(--vscode-editor-background);
    }

    body {
      color: var(--vscode-editor-foreground);
      font-family: var(--vscode-font-family);
    }

    #frame {
      width: 100%;
      height: 100%;
      border: 0;
      background: #101216;
    }

    #loading {
      position: fixed;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background:
        radial-gradient(circle at top left, rgba(57, 113, 177, 0.18), transparent 38%),
        radial-gradient(circle at bottom right, rgba(214, 142, 40, 0.15), transparent 32%),
        var(--vscode-editor-background);
      transition: opacity 180ms ease;
      z-index: 1;
    }

    #loading.hidden {
      opacity: 0;
      pointer-events: none;
    }

    .loading-card {
      padding: 18px 22px;
      border-radius: 12px;
      border: 1px solid var(--vscode-panel-border);
      background: rgba(18, 20, 24, 0.9);
      box-shadow: 0 14px 40px rgba(0, 0, 0, 0.22);
      text-align: center;
      min-width: 220px;
    }

    .loading-card strong {
      display: block;
      margin-bottom: 8px;
      font-size: 14px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      opacity: 0.8;
    }

    .loading-card span {
      font-size: 12px;
      opacity: 0.75;
      word-break: break-all;
    }
  </style>
</head>
<body>
  <div id="loading">
    <div class="loading-card">
      <strong>Loading Netron</strong>
      <span>${escapeHtml(fileName)}</span>
    </div>
  </div>
  <iframe
    id="frame"
    src="${escapeAttribute(iframeSource)}"
    sandbox="allow-forms allow-scripts allow-same-origin allow-downloads"
  ></iframe>
  <script nonce="${nonce}">
    const frame = document.getElementById('frame');
    const loading = document.getElementById('loading');
    const hideLoading = () => loading.classList.add('hidden');
    frame.addEventListener('load', () => {
      window.setTimeout(hideLoading, 120);
    });
  </script>
</body>
</html>`;
}

function getShellHtml(webview, { fileName, body, script }) {
  const nonce = createNonce();
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta
    http-equiv="Content-Security-Policy"
    content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';"
  />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${escapeHtml(fileName)}</title>
  <style>
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
    }

    body {
      display: flex;
      align-items: center;
      justify-content: center;
      background:
        radial-gradient(circle at top left, rgba(57, 113, 177, 0.18), transparent 38%),
        radial-gradient(circle at bottom right, rgba(214, 142, 40, 0.15), transparent 32%),
        var(--vscode-editor-background);
      color: var(--vscode-editor-foreground);
      font-family: var(--vscode-font-family);
      padding: 24px;
      box-sizing: border-box;
    }

    .state-card {
      width: min(720px, 100%);
      border-radius: 14px;
      border: 1px solid var(--vscode-panel-border);
      padding: 24px;
      background: rgba(18, 20, 24, 0.9);
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.2);
    }

    .state-card.error {
      border-color: var(--vscode-errorForeground);
    }

    .state-title {
      font-size: 13px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      opacity: 0.75;
      margin-bottom: 12px;
    }

    .state-copy {
      margin: 0;
      font-size: 13px;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }
  </style>
</head>
<body>
  ${body}
  <script nonce="${nonce}">${script}</script>
</body>
</html>`;
}

function createNonce() {
  return Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function escapeAttribute(value) {
  return escapeHtml(value);
}

function deactivate() {
  for (const session of Array.from(activeSessions)) {
    disposeSession(session);
    activeSessions.delete(session);
  }
}

module.exports = {
  activate,
  deactivate
};
