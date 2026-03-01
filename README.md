# Remote Model Viewer

一个最小可运行的 VS Code 扩展，用来在本地或 Remote SSH 宿主机上直接启动 Netron，并在 VS Code 内嵌显示它的原生 UI。

核心目标：

- 不把权重文件先下载到本地磁盘再打开
- 在 Remote SSH 场景里直接读取远端模型文件
- UI 直接复用 Netron 服务页面，因此交互和视觉基本就是 Netron 本身

## 工作方式

这个扩展不会自己解析模型文件，也不会把 Netron 前端重新实现一遍。

它的做法是：

1. 扩展在当前 extension host 所在机器上启动一个 Python 进程
2. 该进程调用 `netron.start(...)`，直接读取目标模型文件
3. 扩展使用 `vscode.env.asExternalUri()` 把 `127.0.0.1:<port>` 变成 VS Code 可访问的地址
4. 在 webview 里用 `iframe` 打开这个地址

如果你是通过 Remote SSH 打开的工作区，那么 extension host 通常运行在远端机器上。此时 Netron 也会运行在远端，所以模型文件仍然留在远端，不需要先复制到本地磁盘。

## 前置要求

需要在扩展实际运行的那台机器上安装 Python 和 Netron：

```bash
pip install netron
```

在 Remote SSH 模式下，这里的“那台机器”通常是远端 Linux 服务器，不是你的本地 Mac/Windows。

## 使用方式

安装并启动扩展开发宿主后：

- 在资源管理器里右键模型文件，选择 `Open in Remote Model Viewer`
- 或者打开文件后，在编辑器标题栏执行 `Open Active File in Remote Model Viewer`
- 也可以从命令面板运行这两个命令

## 支持说明

模型格式由 Netron 自己决定，常见如：

- `.onnx`
- `.pt`
- `.pth`
- `.pb`
- `.tflite`
- `.mlmodel`
- `.h5`
- `.keras`
- `.safetensors`

这个扩展本身不做格式白名单限制，只要 Netron 能识别，它就能尝试打开。

## 配置项

- `remoteModelViewer.pythonCommand`: 启动 Python 的命令，默认 `python3`
- `remoteModelViewer.startupTimeoutMs`: 等待 Netron 服务可用的超时时间，默认 `15000`
- `remoteModelViewer.enableVerboseServerLog`: 是否启用 `netron.start(..., log=True)` 的详细日志

## 开发

当前实现是零 npm 运行时依赖的 CommonJS 扩展，不需要构建步骤。

仓库已经包含 `.vscode/launch.json`，在 VS Code 里打开这个目录后，直接按 `F5` 就会使用 `Run Extension` 配置启动 `Extension Development Host`。

本地自检：

```bash
npm run check
```

打包分发：

```bash
npm install
npm run package
```

执行后会在当前目录生成一个 `.vsix` 文件，默认文件名类似：

```text
remote-model-viewer-0.0.2.vsix
```

你可以直接把这个 `.vsix` 发给别人安装，或者自己在 VS Code 里用 “Extensions: Install from VSIX...” 安装。

## 已知限制

- 当前版本依赖远端已安装 `netron` Python 包，不会自动安装
- webview 里展示的是 Netron 原生页面，因此扩展层只负责“拉起服务 + 嵌入展示”，不拦截 Netron 内部行为
- “不下载权重”指不额外把模型文件复制到本地磁盘；浏览器/VS Code 与 Netron 服务之间仍然会按页面需要传输可视化数据

## 参考

- [Netron](https://github.com/lutzroeder/netron)
- [vtemplier/vscode-netron](https://github.com/vtemplier/vscode-netron)
- [chingweihsu0809/NetronInVSCode](https://github.com/chingweihsu0809/NetronInVSCode)
