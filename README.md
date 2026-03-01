# Remote Model Viewer

Remote Model Viewer is a VS Code extension for inspecting machine learning model files without leaving the editor.  
Remote Model Viewer 是一个用于在 VS Code 内查看机器学习模型文件的扩展。

It provides two viewers:  
它提供两种查看方式：

- `Remote Model Viewer`: launches [Netron](https://github.com/lutzroeder/netron) on the current extension host and embeds the Netron UI inside VS Code.  
  `Remote Model Viewer`：在当前 extension host 所在机器上启动 [Netron](https://github.com/lutzroeder/netron)，并将 Netron UI 嵌入到 VS Code 中。
- `Safetensors Explorer`: a built-in viewer for `.safetensors` and `.safetensors.index.json` with tensor browsing and paged value preview.  
  `Safetensors Explorer`：内置的 `.safetensors` / `.safetensors.index.json` 查看器，支持 tensor 浏览和分页预览。

It works both locally and with Remote SSH.  
它既支持本地工作区，也支持 Remote SSH 工作区。

## Features / 功能

- Open model files in an embedded Netron view from the Explorer, editor title, or tab context menu.  
  可从资源管理器、编辑器标题栏或标签页右键菜单中，以内嵌 Netron 方式打开模型文件。
- Open `.safetensors` and `.safetensors.index.json` in a dedicated explorer.  
  可使用专用查看器打开 `.safetensors` 和 `.safetensors.index.json`。
- Browse tensors by hierarchical name prefix.  
  支持按 tensor name 前缀层级浏览。
- Filter tensors by keyword or prefix-only matching.  
  支持关键字过滤和前缀过滤。
- Inspect tensor metadata such as shape, dtype, shard, element count, and byte size.  
  可查看 tensor 的 shape、dtype、分片、元素数量和字节大小等元信息。
- Preview tensor values with manual offsets and paged loading.  
  支持手动 offset 和分页加载的 tensor 值预览。

## Requirements / 依赖要求

For the Netron-based viewer:  
对于基于 Netron 的查看器：

- Python must be available on the extension host.  
  extension host 所在机器需要有 Python。
- The `netron` Python package must be installed on the extension host.  
  extension host 所在机器需要安装 `netron` Python 包。

```bash
pip install netron
```

In a Remote SSH session, install `netron` on the remote server, not on the local machine.  
在 Remote SSH 场景下，应当在远程服务器上安装 `netron`，而不是本地机器。

The Safetensors Explorer does not depend on Python or the `netron` package.  
`Safetensors Explorer` 不依赖 Python，也不依赖 `netron`。

## Commands / 命令

This extension contributes the following commands:  
此扩展提供以下命令：

- `Open in Remote Model Viewer`
- `Open Active File in Remote Model Viewer`
- `Open in Safetensors Explorer`

These commands are available from:  
这些命令可以从以下位置使用：

- Explorer context menu  
  资源管理器右键菜单
- Editor title actions  
  编辑器标题栏
- Editor tab context menu  
  编辑器标签页右键菜单
- Command Palette  
  命令面板

## Usage / 使用方式

### Open With Netron / 使用 Netron 打开

Use `Open in Remote Model Viewer` for model formats supported by Netron, for example:  
对于 Netron 支持的模型格式，可使用 `Open in Remote Model Viewer`，例如：

- `.onnx`
- `.pt`
- `.pth`
- `.pb`
- `.tflite`
- `.mlmodel`
- `.h5`
- `.keras`
- `.safetensors`

The extension starts Netron on the extension host, exposes the local service through `vscode.env.asExternalUri()`, and renders the Netron page inside a webview.  
扩展会在 extension host 所在机器启动 Netron，通过 `vscode.env.asExternalUri()` 暴露本地服务，并在 webview 中渲染 Netron 页面。

### Open With Safetensors Explorer / 使用 Safetensors Explorer 打开

Use `Open in Safetensors Explorer` for:  
对于以下文件，可使用 `Open in Safetensors Explorer`：

- `.safetensors`
- `.safetensors.index.json`

The Safetensors Explorer provides:  
`Safetensors Explorer` 提供：

- Tensor tree grouped by name prefix  
  按名称前缀分组的 tensor 树
- Tensor metadata panel  
  tensor 元信息面板
- Offset-based value preview  
  基于 offset 的值预览
- `Load Tensor Values` and `Load More` pagination  
  `Load Tensor Values` 与 `Load More` 分页加载

Tensor values are loaded only when requested from the detail pane.  
只有在详情面板里主动请求时，才会读取 tensor 值。

## Remote SSH Behavior / Remote SSH 行为

When used with Remote SSH:  
在 Remote SSH 场景下：

- The extension host runs on the remote machine.  
  extension host 运行在远程机器上。
- Netron is launched on the remote machine.  
  Netron 运行在远程机器上。
- Safetensors are read directly from the remote filesystem.  
  Safetensors 文件直接从远程文件系统读取。
- You do not need to manually copy the model file to local disk before opening it.  
  不需要先手动把模型文件复制到本地磁盘再打开。

For the Netron viewer, the UI is still rendered in the local VS Code client.  
对于 Netron 查看器，UI 仍然是在本地 VS Code 客户端中渲染的。

## Extension Settings / 扩展配置

This extension contributes the following settings:  
此扩展提供以下配置项：

- `remoteModelViewer.pythonCommand`: Python executable used to launch the Netron helper process. Default: `python3`  
  `remoteModelViewer.pythonCommand`：启动 Netron 辅助进程所使用的 Python 命令，默认值为 `python3`
- `remoteModelViewer.startupTimeoutMs`: Timeout for waiting until Netron becomes reachable. Default: `15000`  
  `remoteModelViewer.startupTimeoutMs`：等待 Netron 可访问的超时时间，默认值为 `15000`
- `remoteModelViewer.enableVerboseServerLog`: Enables verbose logging when starting Netron. Default: `false`  
  `remoteModelViewer.enableVerboseServerLog`：启动 Netron 时启用详细日志，默认值为 `false`

## Installation / 安装

Install from a packaged `.vsix`:  
通过打包后的 `.vsix` 安装：

1. Run:  
   执行：

```bash
npm install
npm run package
```

2. In VS Code, run `Extensions: Install from VSIX...`  
   在 VS Code 中执行 `Extensions: Install from VSIX...`
3. Select the generated `.vsix` file, for example:  
   选择生成的 `.vsix` 文件，例如：

```text
remote-model-viewer-0.0.6.vsix
```

## Development / 开发

This repository is a plain CommonJS extension and does not require a build step.  
这个仓库是一个普通的 CommonJS 扩展，不需要额外构建步骤。

Useful commands:  
常用命令：

```bash
npm run check
```

```bash
npm run package
```

To launch an Extension Development Host:  
启动扩展开发宿主的方法：

1. Open this folder in VS Code  
   用 VS Code 打开当前目录
2. Press `F5`  
   按 `F5`
3. Choose the `Run Extension` configuration if prompted  
   如果出现配置选择，使用 `Run Extension`

## Known Limitations / 已知限制

- The Netron viewer depends on the upstream `netron` Python package and does not bundle it.  
  Netron 查看器依赖上游 `netron` Python 包，本扩展不会内置它。
- The extension embeds the Netron UI rather than re-implementing it.  
  扩展是嵌入 Netron UI，而不是重写 Netron。
- `Safetensors Explorer` focuses on metadata browsing and paged preview, not full tensor export.  
  `Safetensors Explorer` 目前侧重元信息浏览和分页预览，不是完整 tensor 导出工具。
- Large safetensors files or large sharded indexes may still take noticeable time to open.  
  较大的 safetensors 文件或大型分片索引在打开时仍可能有明显延迟。
- Tensor preview currently targets common dtypes such as `BOOL`, integer types, `F16`, `BF16`, `F32`, and `F64`.  
  当前 tensor 预览主要支持常见 dtype，例如 `BOOL`、整数类型、`F16`、`BF16`、`F32` 和 `F64`。

## Related Projects / 相关项目

- [Netron](https://github.com/lutzroeder/netron)
- [vtemplier/vscode-netron](https://github.com/vtemplier/vscode-netron)
- [NetronInVSCode](https://github.com/chingweihsu0809/NetronInVSCode)
