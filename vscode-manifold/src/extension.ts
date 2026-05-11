import * as vscode from 'vscode';
import * as https from 'https';
import * as http from 'http';

interface ManifoldResult {
  action: string;
  risk_score: number;
  domain: string;
  vetoed: boolean;
}

async function checkWithManifold(
  text: string,
  serverUrl: string,
  apiKey: string,
): Promise<ManifoldResult> {
  const body = JSON.stringify({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: text }],
  });
  return new Promise((resolve, reject) => {
    const url = new URL('/v1/chat/completions', serverUrl);
    const lib = url.protocol === 'https:' ? https : http;
    const req = lib.request(
      {
        hostname: url.hostname,
        port: url.port || (url.protocol === 'https:' ? 443 : 80),
        path: url.pathname,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
          'Content-Length': Buffer.byteLength(body),
        },
      },
      (res) => {
        let data = '';
        res.on('data', (chunk) => {
          data += chunk;
        });
        res.on('end', () => {
          try {
            const parsed = JSON.parse(data);
            const manifold = parsed._manifold || {};
            resolve({
              action: manifold.action || 'unknown',
              risk_score: manifold.risk_score || 0,
              domain: manifold.domain || 'general',
              vetoed: manifold.vetoed || false,
            });
          } catch {
            reject(new Error('Invalid response'));
          }
        });
      },
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

export function activate(context: vscode.ExtensionContext): void {
  const config = () => vscode.workspace.getConfiguration('manifold');

  // Command: check selected text
  const checkCmd = vscode.commands.registerCommand(
    'manifold.checkSelection',
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        return;
      }
      const selection = editor.document.getText(editor.selection);
      if (!selection.trim()) {
        vscode.window.showWarningMessage('Select some code or text to check.');
        return;
      }
      const serverUrl = config().get<string>('serverUrl', 'http://localhost:8080');
      const apiKey = config().get<string>('apiKey', '');
      if (!apiKey) {
        vscode.window.showErrorMessage(
          'MANIFOLD: Set your API key in settings (manifold.apiKey).',
        );
        return;
      }
      try {
        const result = await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: 'MANIFOLD: Checking risk...',
          },
          () => checkWithManifold(selection, serverUrl, apiKey),
        );
        const icon = result.vetoed ? '🔴' : result.risk_score > 0.6 ? '🟡' : '🟢';
        const msg =
          `${icon} Risk: ${(result.risk_score * 100).toFixed(0)}% · ` +
          `Action: ${result.action} · Domain: ${result.domain}`;
        if (result.vetoed) {
          vscode.window.showWarningMessage(`MANIFOLD blocked: ${msg}`);
        } else {
          vscode.window.showInformationMessage(`MANIFOLD: ${msg}`);
        }
      } catch (err) {
        vscode.window.showErrorMessage(`MANIFOLD unreachable: ${err}`);
      }
    },
  );

  // Command: open dashboard
  const dashCmd = vscode.commands.registerCommand('manifold.openDashboard', () => {
    const url = config().get<string>('serverUrl', 'http://localhost:8080');
    void vscode.env.openExternal(vscode.Uri.parse(`${url}/report`));
  });

  // Auto-check on save if enabled
  const saveListener = vscode.workspace.onDidSaveTextDocument(async (doc) => {
    if (!config().get<boolean>('enabled', true)) {
      return;
    }
    if (!config().get<boolean>('autoCheck', false)) {
      return;
    }
    const apiKey = config().get<string>('apiKey', '');
    if (!apiKey) {
      return;
    }
    const text = doc.getText();
    if (text.length < 20 || text.length > 4000) {
      return;
    }
    const serverUrl = config().get<string>('serverUrl', 'http://localhost:8080');
    try {
      const result = await checkWithManifold(text.slice(0, 500), serverUrl, apiKey);
      if (result.vetoed || result.risk_score > 0.75) {
        vscode.window.showWarningMessage(
          `MANIFOLD: High-risk save detected (${(result.risk_score * 100).toFixed(0)}%). ` +
            `Action: ${result.action}`,
        );
      }
    } catch {
      // silent fail — don't interrupt workflow
    }
  });

  context.subscriptions.push(checkCmd, dashCmd, saveListener);
}

export function deactivate(): void {}
