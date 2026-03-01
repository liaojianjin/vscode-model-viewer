'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');

const HEADER_LIMIT_BYTES = 100 * 1024 * 1024;
const PREVIEW_PAGE_ELEMENT_LIMIT = 256;
const MAX_PREVIEW_BYTES = 256 * 1024;

function registerSafetensorsExplorer(output) {
  const vscode = require('vscode');
  return vscode.commands.registerCommand('remoteModelViewer.openSafetensorsExplorer', async (uri) => {
    await openSafetensorsExplorer(output, uri);
  });
}

async function openSafetensorsExplorer(output, targetUri) {
  const vscode = require('vscode');
  const uri = targetUri ?? vscode.window.activeTextEditor?.document?.uri;
  if (!uri) {
    vscode.window.showErrorMessage('No file selected to open in Safetensors Explorer.');
    return;
  }

  try {
    const stat = await vscode.workspace.fs.stat(uri);
    if ((stat.type & vscode.FileType.File) === 0) {
      vscode.window.showErrorMessage('Safetensors Explorer can only open files.');
      return;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    vscode.window.showErrorMessage(`Unable to access the selected file: ${message}`);
    return;
  }

  const filePath = toHostFilePath(uri);
  const fileName = path.basename(filePath);

  if (!isSupportedSafetensorsInput(filePath)) {
    vscode.window.showErrorMessage('Safetensors Explorer only supports .safetensors and .safetensors.index.json files.');
    return;
  }

  let repository;
  try {
    repository = await SafetensorsRepository.open(filePath);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    output.appendLine(`[error] ${message}`);
    vscode.window.showErrorMessage(`Unable to open ${fileName} in Safetensors Explorer. See "Remote Model Viewer" output for details.`);
    return;
  }

  const panel = vscode.window.createWebviewPanel(
    'remoteModelViewer.safetensorsExplorer',
    `Safetensors: ${fileName}`,
    vscode.ViewColumn.Active,
    {
      enableScripts: true,
      retainContextWhenHidden: true
    }
  );

  panel.webview.html = getSafetensorsExplorerHtml(panel.webview, fileName);

  panel.webview.onDidReceiveMessage(async (message) => {
    try {
      switch (message?.type) {
        case 'ready': {
          await panel.webview.postMessage({
            type: 'init',
            payload: await repository.getManifestPayload()
          });
          break;
        }
        case 'select-tensor': {
          if (typeof message.name !== 'string') {
            return;
          }
          await panel.webview.postMessage({
            type: 'tensor-details',
            payload: await repository.getTensorDetails(message.name)
          });
          break;
        }
        case 'load-values': {
          if (typeof message.name !== 'string') {
            return;
          }
          const offset = Number.isFinite(message.offset) ? Math.max(0, Math.trunc(message.offset)) : 0;
          const limit = Number.isFinite(message.limit) ? Math.max(1, Math.trunc(message.limit)) : PREVIEW_PAGE_ELEMENT_LIMIT;
          await panel.webview.postMessage({
            type: 'tensor-preview',
            payload: await repository.getTensorPreview(message.name, { offset, limit })
          });
          break;
        }
        default: {
          break;
        }
      }
    } catch (error) {
      const messageText = error instanceof Error ? error.message : String(error);
      output.appendLine(`[error] ${messageText}`);
      await panel.webview.postMessage({
        type: 'error',
        payload: {
          message: messageText
        }
      });
    }
  });
}

class SafetensorsRepository {
  static async open(filePath) {
    if (isSafetensorsIndexPath(filePath)) {
      return await SafetensorsRepository.fromIndex(filePath);
    }
    if (isSafetensorsPath(filePath)) {
      return await SafetensorsRepository.fromFile(filePath);
    }
    throw new Error('Unsupported file type.');
  }

  static async fromFile(filePath) {
    const header = await readSafetensorsHeader(filePath, path.basename(filePath));
    return new SafetensorsRepository({
      mode: 'single',
      rootPath: filePath,
      header,
      indexMetadata: header.metadata
    });
  }

  static async fromIndex(filePath) {
    const text = await fs.readFile(filePath, 'utf8');
    let json;
    try {
      json = JSON.parse(text);
    } catch (error) {
      throw new Error(`Failed to parse ${path.basename(filePath)}: ${error instanceof Error ? error.message : String(error)}`);
    }

    if (!json || typeof json !== 'object' || !json.weight_map || typeof json.weight_map !== 'object') {
      throw new Error(`Unsupported safetensors index file: ${path.basename(filePath)}`);
    }

    const weightMap = new Map();
    const shardNames = new Set();
    for (const [tensorName, shardName] of Object.entries(json.weight_map)) {
      if (typeof shardName !== 'string') {
        continue;
      }
      weightMap.set(tensorName, shardName);
      shardNames.add(shardName);
    }

    return new SafetensorsRepository({
      mode: 'index',
      rootPath: filePath,
      weightMap,
      shardCount: shardNames.size,
      indexMetadata: json.metadata && typeof json.metadata === 'object' ? json.metadata : null
    });
  }

  constructor({ mode, rootPath, header = null, weightMap = null, shardCount = 0, indexMetadata = null }) {
    this.mode = mode;
    this.rootPath = rootPath;
    this.header = header;
    this.weightMap = weightMap;
    this.shardCount = shardCount;
    this.indexMetadata = indexMetadata;
    this.headerCache = new Map();
    this.recordIndex = new Map();
    this.allRecords = null;

    if (header) {
      this.headerCache.set(rootPath, header);
      for (const record of header.records.values()) {
        this.recordIndex.set(record.name, record);
      }
    }
  }

  async getManifestPayload() {
    const tensors = (await this.getAllRecords()).map((record) => toTensorSummary(record));

    return {
      title: path.basename(this.rootPath),
      mode: this.mode,
      tensorCount: tensors.length,
      shardCount: this.mode === 'index' ? this.shardCount : 1,
      metadata: this.indexMetadata,
      metadataEntries: toMetadataEntries(this.indexMetadata),
      tensors
    };
  }

  async getTensorDetails(name) {
    const record = await this.getRecord(name);
    return {
      tensor: toTensorSummary(record)
    };
  }

  async getTensorPreview(name, options = {}) {
    const record = await this.getRecord(name);
    const preview = await readTensorPreview(record, options);
    return {
      name: record.name,
      preview
    };
  }

  async getRecord(name) {
    if (this.recordIndex.has(name)) {
      return this.recordIndex.get(name);
    }

    if (this.mode === 'single') {
      throw new Error(`Tensor not found: ${name}`);
    }

    const sourceFile = this.weightMap.get(name);
    if (!sourceFile) {
      throw new Error(`Tensor not found in index: ${name}`);
    }

    const resolvedPath = path.join(path.dirname(this.rootPath), sourceFile);
    const header = await this.loadHeader(resolvedPath, sourceFile);
    const record = header.records.get(name);
    if (!record) {
      throw new Error(`Tensor "${name}" was mapped to ${sourceFile}, but no matching entry was found in that shard.`);
    }
    return record;
  }

  async getAllRecords() {
    if (this.allRecords) {
      return this.allRecords;
    }

    if (this.mode === 'index') {
      const shardNames = Array.from(new Set(this.weightMap.values()));
      await Promise.all(
        shardNames.map(async (sourceFile) => {
          const resolvedPath = path.join(path.dirname(this.rootPath), sourceFile);
          await this.loadHeader(resolvedPath, sourceFile);
        })
      );
    }

    this.allRecords = sortedRecords(this.recordIndex.values());
    return this.allRecords;
  }

  async loadHeader(filePath, sourceFile) {
    if (!this.headerCache.has(filePath)) {
      const header = await readSafetensorsHeader(filePath, sourceFile);
      this.headerCache.set(filePath, header);
      for (const record of header.records.values()) {
        this.recordIndex.set(record.name, record);
      }
    }
    return this.headerCache.get(filePath);
  }
}

async function readSafetensorsHeader(filePath, sourceFile) {
  const handle = await fs.open(filePath, 'r');

  try {
    const prefix = Buffer.alloc(8);
    const prefixRead = await handle.read(prefix, 0, 8, 0);
    if (prefixRead.bytesRead !== 8) {
      throw new Error(`File is too small to be a safetensors container: ${path.basename(filePath)}`);
    }

    const headerByteLengthBigInt = prefix.readBigUInt64LE(0);
    if (headerByteLengthBigInt > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(`Header is too large to parse safely: ${path.basename(filePath)}`);
    }

    const headerByteLength = Number(headerByteLengthBigInt);
    if (headerByteLength <= 1 || headerByteLength > HEADER_LIMIT_BYTES) {
      throw new Error(`Unexpected safetensors header size (${headerByteLength}) in ${path.basename(filePath)}`);
    }

    const buffer = Buffer.alloc(headerByteLength);
    const headerRead = await handle.read(buffer, 0, headerByteLength, 8);
    if (headerRead.bytesRead !== headerByteLength) {
      throw new Error(`Unexpected end of file while reading safetensors header: ${path.basename(filePath)}`);
    }

    let json;
    try {
      json = JSON.parse(buffer.toString('utf8'));
    } catch (error) {
      throw new Error(`Failed to parse safetensors header in ${path.basename(filePath)}: ${error instanceof Error ? error.message : String(error)}`);
    }

    const dataStart = 8 + headerByteLength;
    const records = new Map();
    for (const [name, value] of Object.entries(json)) {
      if (name === '__metadata__') {
        continue;
      }

      if (!value || typeof value !== 'object' || typeof value.dtype !== 'string' || !Array.isArray(value.shape) || !Array.isArray(value.data_offsets) || value.data_offsets.length !== 2) {
        continue;
      }

      const start = toNonNegativeNumber(value.data_offsets[0]);
      const end = toNonNegativeNumber(value.data_offsets[1]);
      if (start === null || end === null || end < start) {
        continue;
      }

      const shape = value.shape.map((dimension) => (typeof dimension === 'number' ? dimension : Number(dimension)));
      const record = {
        name,
        dtype: value.dtype,
        shape,
        sourceFile,
        filePath,
        byteLength: end - start,
        dataOffset: dataStart + start,
        elementCount: getElementCount(shape)
      };
      records.set(name, record);
    }

    return {
      sourceFile,
      filePath,
      metadata: json.__metadata__ && typeof json.__metadata__ === 'object' ? json.__metadata__ : null,
      records
    };
  } finally {
    await handle.close();
  }
}

async function readTensorPreview(record, options = {}) {
  const decoder = getDecoder(record.dtype);
  if (!decoder) {
    return {
      supported: false,
      reason: `Tensor preview for dtype "${record.dtype}" is not implemented yet.`
    };
  }

  const offset = Number.isFinite(options.offset) ? Math.max(0, Math.trunc(options.offset)) : 0;
  const limit = Number.isFinite(options.limit) ? Math.max(1, Math.trunc(options.limit)) : PREVIEW_PAGE_ELEMENT_LIMIT;
  const totalElements = record.elementCount;
  const totalElementNumber = totalElements !== null && totalElements <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(totalElements) : null;
  const bytesPerElement = decoder.bytesPerElement;
  const maxElementsPerPage = Math.max(1, Math.floor(MAX_PREVIEW_BYTES / bytesPerElement));

  if (totalElementNumber !== null && offset >= totalElementNumber) {
    return {
      supported: true,
      dtype: record.dtype,
      offset,
      values: [],
      loadedElements: 0,
      nextOffset: offset,
      totalElements: formatElementCount(totalElements),
      hasMore: false
    };
  }

  const availableElementsFromBytes = Math.max(0, Math.floor(Math.max(0, record.byteLength - (offset * bytesPerElement)) / bytesPerElement));
  const remainingElements = totalElementNumber === null ? availableElementsFromBytes : Math.max(0, totalElementNumber - offset);
  const requestedElements = Math.max(0, Math.min(remainingElements, availableElementsFromBytes, limit, maxElementsPerPage));
  const byteLength = requestedElements * bytesPerElement;
  const buffer = byteLength > 0 ? await readFileRange(record.filePath, record.dataOffset + (offset * bytesPerElement), byteLength) : Buffer.alloc(0);
  const values = decodeValues(buffer, decoder);
  const nextOffset = offset + values.length;
  const hasMore = totalElementNumber === null ?
    (record.byteLength > nextOffset * bytesPerElement) :
    (nextOffset < totalElementNumber);

  return {
    supported: true,
    dtype: record.dtype,
    offset,
    values,
    loadedElements: values.length,
    nextOffset,
    totalElements: formatElementCount(totalElements),
    hasMore
  };
}

async function readFileRange(filePath, offset, length) {
  const handle = await fs.open(filePath, 'r');
  try {
    const buffer = Buffer.alloc(length);
    const { bytesRead } = await handle.read(buffer, 0, length, offset);
    return buffer.subarray(0, bytesRead);
  } finally {
    await handle.close();
  }
}

function decodeValues(buffer, decoder) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const values = [];
  for (let offset = 0; offset + decoder.bytesPerElement <= buffer.byteLength; offset += decoder.bytesPerElement) {
    values.push(decoder.read(view, offset));
  }
  return values;
}

function getDecoder(dtype) {
  const decoders = {
    BOOL: { bytesPerElement: 1, read: (view, offset) => view.getUint8(offset) !== 0 },
    U8: { bytesPerElement: 1, read: (view, offset) => view.getUint8(offset) },
    I8: { bytesPerElement: 1, read: (view, offset) => view.getInt8(offset) },
    U16: { bytesPerElement: 2, read: (view, offset) => view.getUint16(offset, true) },
    I16: { bytesPerElement: 2, read: (view, offset) => view.getInt16(offset, true) },
    U32: { bytesPerElement: 4, read: (view, offset) => view.getUint32(offset, true) },
    I32: { bytesPerElement: 4, read: (view, offset) => view.getInt32(offset, true) },
    U64: { bytesPerElement: 8, read: (view, offset) => view.getBigUint64(offset, true).toString() },
    I64: { bytesPerElement: 8, read: (view, offset) => view.getBigInt64(offset, true).toString() },
    F64: { bytesPerElement: 8, read: (view, offset) => view.getFloat64(offset, true) },
    F32: { bytesPerElement: 4, read: (view, offset) => view.getFloat32(offset, true) },
    F16: { bytesPerElement: 2, read: (view, offset) => decodeFloat16(view.getUint16(offset, true)) },
    BF16: { bytesPerElement: 2, read: (view, offset) => decodeBFloat16(view.getUint16(offset, true)) }
  };
  return decoders[dtype] || null;
}

function decodeFloat16(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;

  if (exponent === 0) {
    if (fraction === 0) {
      return sign === 1 ? 0 : -0;
    }
    return sign * (fraction / 1024) * Math.pow(2, -14);
  }

  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : NaN;
  }

  return sign * (1 + (fraction / 1024)) * Math.pow(2, exponent - 15);
}

function decodeBFloat16(value) {
  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  view.setUint32(0, value << 16, true);
  return view.getFloat32(0, true);
}

function sortedRecords(records) {
  return Array.from(records).sort((left, right) => left.name.localeCompare(right.name));
}

function toTensorSummary(record) {
  return {
    name: record.name,
    dtype: record.dtype,
    shape: record.shape,
    shapeText: formatShape(record.shape),
    sourceFile: record.sourceFile,
    byteLength: record.byteLength,
    totalElements: formatElementCount(record.elementCount),
    metadataReady: true,
    previewSupported: Boolean(getDecoder(record.dtype))
  };
}

function formatShape(shape) {
  if (!Array.isArray(shape) || shape.length === 0) {
    return '[]';
  }
  return `[${shape.map((dimension) => String(dimension)).join(', ')}]`;
}

function formatElementCount(value) {
  return value === null ? '?' : value.toString();
}

function toMetadataEntries(metadata) {
  if (!metadata || typeof metadata !== 'object') {
    return [];
  }

  return Object.entries(metadata)
    .map(([key, value]) => ({
      key,
      value: typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? String(value) : JSON.stringify(value)
    }))
    .sort((left, right) => left.key.localeCompare(right.key));
}

function getElementCount(shape) {
  if (!Array.isArray(shape)) {
    return null;
  }

  let count = 1n;
  for (const dimension of shape) {
    if (!Number.isFinite(dimension) || dimension < 0) {
      return null;
    }
    count *= BigInt(Math.trunc(dimension));
  }
  return count;
}

function toNonNegativeNumber(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return null;
  }
  return Math.trunc(value);
}

function isSupportedSafetensorsInput(filePath) {
  return isSafetensorsPath(filePath) || isSafetensorsIndexPath(filePath);
}

function isSafetensorsPath(filePath) {
  return filePath.endsWith('.safetensors');
}

function isSafetensorsIndexPath(filePath) {
  return filePath.endsWith('.safetensors.index.json');
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

function getSafetensorsExplorerHtml(webview, fileName) {
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
    :root {
      color-scheme: light dark;
    }

    * {
      box-sizing: border-box;
    }

    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: var(--vscode-editor-background);
      color: var(--vscode-editor-foreground);
      font-family: var(--vscode-font-family);
      font-size: 13px;
    }

    body {
      display: grid;
      grid-template-rows: auto 1fr;
      background:
        radial-gradient(circle at top left, rgba(21, 105, 167, 0.16), transparent 36%),
        radial-gradient(circle at bottom right, rgba(218, 145, 34, 0.12), transparent 30%),
        var(--vscode-editor-background);
    }

    .toolbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--vscode-panel-border);
      background: rgba(18, 20, 24, 0.9);
    }

    .toolbar-title {
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 0;
    }

    .toolbar-title strong {
      font-size: 14px;
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .toolbar-title span {
      color: var(--vscode-descriptionForeground);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .badge-row {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }

    .badge {
      border: 1px solid var(--vscode-panel-border);
      border-radius: 999px;
      padding: 4px 10px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--vscode-descriptionForeground);
      font-size: 12px;
    }

    .layout {
      min-height: 0;
      display: grid;
      grid-template-columns: 1fr;
      grid-template-rows: minmax(0, 3fr) minmax(0, 2fr);
      gap: 1px;
      background: var(--vscode-panel-border);
    }

    .pane {
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      background: var(--vscode-editor-background);
    }

    .pane-scroll {
      width: 100%;
      height: 100%;
      overflow: auto;
      padding: 14px;
    }

    .stack {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .card {
      border: 1px solid var(--vscode-panel-border);
      border-radius: 12px;
      background: rgba(18, 20, 24, 0.7);
      padding: 12px;
    }

    .section-title {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--vscode-descriptionForeground);
      margin-bottom: 10px;
    }

    .filter-grid {
      display: grid;
      gap: 10px;
    }

    .filter-input {
      width: 100%;
      border: 1px solid var(--vscode-input-border, var(--vscode-panel-border));
      border-radius: 8px;
      padding: 8px 10px;
      background: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
      font: inherit;
    }

    .filter-toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--vscode-descriptionForeground);
      user-select: none;
    }

    .filter-toggle input {
      margin: 0;
    }

    .metadata-grid {
      display: grid;
      grid-template-columns: max-content 1fr;
      gap: 8px 12px;
    }

    .metadata-grid dt {
      color: var(--vscode-descriptionForeground);
    }

    .metadata-grid dd {
      margin: 0;
      word-break: break-all;
    }

    .tree-empty,
    .detail-empty,
    .error {
      border: 1px solid var(--vscode-panel-border);
      border-radius: 12px;
      padding: 16px;
      background: rgba(18, 20, 24, 0.7);
      color: var(--vscode-descriptionForeground);
      line-height: 1.5;
    }

    .error {
      border-color: var(--vscode-errorForeground);
      color: var(--vscode-errorForeground);
      white-space: pre-wrap;
    }

    .tree {
      border: 1px solid var(--vscode-panel-border);
      border-radius: 12px;
      overflow: hidden;
      background: rgba(8, 18, 38, 0.45);
    }

    .tree-header,
    .tree-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(132px, 180px) minmax(88px, 116px);
      gap: 12px;
      align-items: center;
      min-width: 0;
    }

    .tree-header {
      padding: 7px 10px;
      border-bottom: 1px solid var(--vscode-panel-border);
      background: linear-gradient(90deg, rgba(32, 70, 170, 0.7), rgba(8, 26, 60, 0.92));
      color: rgba(255, 255, 255, 0.9);
      font-weight: 600;
    }

    .tree-body {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .tree details {
      margin-left: 12px;
    }

    .tree details > summary {
      list-style: none;
      cursor: pointer;
      border-radius: 10px;
      padding: 0;
    }

    .tree details > summary::-webkit-details-marker {
      display: none;
    }

    .tree details > summary > .tree-row:hover {
      background: rgba(255, 255, 255, 0.04);
    }

    .tensor-leaf {
      width: 100%;
      text-align: left;
      padding: 0;
      border: 0;
      background: transparent;
      color: inherit;
      cursor: pointer;
      font: inherit;
    }

    .tensor-leaf:hover .tree-row {
      background: rgba(255, 255, 255, 0.04);
    }

    .tensor-leaf.active .tree-row,
    .tree details.selected > summary > .tree-row {
      background: rgba(21, 105, 167, 0.22);
      color: var(--vscode-foreground);
    }

    .tree-row {
      width: 100%;
      padding: 7px 10px;
      border-radius: 10px;
      color: inherit;
    }

    .tree-name-cell,
    .tree-shape-cell,
    .tree-precision-cell {
      min-width: 0;
    }

    .tree-name-text {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 2px;
      min-width: 0;
      line-height: 1.35;
      text-align: left;
      word-break: break-all;
    }

    .tree-path-prefix {
      color: var(--vscode-descriptionForeground);
      opacity: 0.6;
    }

    .tree-path-name {
      color: var(--vscode-foreground);
      font-weight: 500;
    }

    .tree-group-meta {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      margin-left: 4px;
      color: var(--vscode-descriptionForeground);
      opacity: 0.85;
      white-space: nowrap;
    }

    .tree-caret {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 0.8em;
      transition: transform 120ms ease;
      transform-origin: center;
    }

    .tree details[open] > summary .tree-caret {
      transform: rotate(90deg);
    }

    .tree-shape-cell,
    .tree-precision-cell {
      color: var(--vscode-descriptionForeground);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      white-space: nowrap;
      text-align: left;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .tensor-name {
      font-weight: 600;
      margin-bottom: 4px;
      word-break: break-all;
    }

    .detail-grid {
      display: grid;
      grid-template-columns: max-content 1fr;
      gap: 8px 12px;
      margin: 12px 0 16px;
      padding: 14px;
      border: 1px solid var(--vscode-panel-border);
      border-radius: 12px;
      background: rgba(18, 20, 24, 0.6);
    }

    .detail-grid dt {
      color: var(--vscode-descriptionForeground);
    }

    .detail-grid dd {
      margin: 0;
      word-break: break-all;
    }

    .actions {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }

    .button {
      border: 1px solid var(--vscode-button-border, transparent);
      border-radius: 8px;
      padding: 8px 12px;
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
      cursor: pointer;
      font: inherit;
    }

    .button.secondary {
      background: transparent;
      color: var(--vscode-button-secondaryForeground);
      border-color: var(--vscode-panel-border);
    }

    .button:disabled {
      opacity: 0.5;
      cursor: default;
    }

    .button-content {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }

    .button-icon {
      display: inline-flex;
      width: 14px;
      height: 14px;
      align-items: center;
      justify-content: center;
      flex: 0 0 auto;
    }

    .button-icon svg,
    .tree-caret svg {
      width: 100%;
      height: 100%;
      display: block;
      stroke: currentColor;
      fill: none;
      stroke-width: 1.8;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    .hint {
      color: var(--vscode-descriptionForeground);
    }

    pre {
      margin: 0;
      padding: 14px;
      border: 1px solid var(--vscode-panel-border);
      border-radius: 12px;
      background: rgba(18, 20, 24, 0.7);
      overflow: auto;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .preview-meta {
      margin-bottom: 10px;
      color: var(--vscode-descriptionForeground);
    }

    .offset-panel {
      margin-bottom: 14px;
      padding: 12px;
      border: 1px solid var(--vscode-panel-border);
      border-radius: 12px;
      background: rgba(18, 20, 24, 0.55);
    }

    .offset-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      gap: 10px;
      margin-bottom: 10px;
    }

    .offset-field {
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 0;
    }

    .offset-field label {
      color: var(--vscode-descriptionForeground);
      font-size: 12px;
    }

    .offset-field input {
      width: 100%;
      border: 1px solid var(--vscode-input-border, var(--vscode-panel-border));
      border-radius: 8px;
      padding: 7px 9px;
      background: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
      font: inherit;
    }

    .offset-caption {
      color: var(--vscode-descriptionForeground);
      font-size: 12px;
      line-height: 1.45;
    }

    @media (max-width: 900px) {
      .layout {
        grid-template-rows: minmax(0, 1fr) minmax(220px, auto);
      }
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <div class="toolbar-title">
      <strong id="title">${escapeHtml(fileName)}</strong>
      <span id="subtitle">Reading safetensors metadata from the extension host...</span>
    </div>
    <div class="badge-row">
      <div class="badge" id="tensor-count">0 tensors</div>
      <div class="badge" id="shard-count">1 shard</div>
    </div>
  </div>
  <div class="layout">
    <section class="pane">
      <div class="pane-scroll">
        <div class="stack">
          <div class="card">
            <div class="section-title">Filter</div>
            <div class="filter-grid">
              <input id="search-input" class="filter-input" type="search" placeholder="Search tensor names" />
              <label class="filter-toggle">
                <input id="prefix-toggle" type="checkbox" />
                <span>Prefix only</span>
              </label>
            </div>
          </div>
          <div id="metadata-root"></div>
          <div id="tree-root" class="tree-empty">Loading tensor list...</div>
        </div>
      </div>
    </section>
    <section class="pane">
      <div class="pane-scroll">
        <div id="detail-root" class="detail-empty">Select a tensor from the left tree. Values are loaded on demand from the remote host only when you ask for them.</div>
        <div id="error-root" style="display:none;"></div>
      </div>
    </section>
  </div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const state = {
      manifest: null,
      selectedName: null,
      details: new Map(),
      previews: new Map(),
      tensorIndex: new Map(),
      offsets: new Map(),
      filterText: '',
      prefixOnly: false,
      openGroups: new Set()
    };

    const title = document.getElementById('title');
    const subtitle = document.getElementById('subtitle');
    const tensorCount = document.getElementById('tensor-count');
    const shardCount = document.getElementById('shard-count');
    const metadataRoot = document.getElementById('metadata-root');
    const searchInput = document.getElementById('search-input');
    const prefixToggle = document.getElementById('prefix-toggle');
    const treeRoot = document.getElementById('tree-root');
    const detailRoot = document.getElementById('detail-root');
    const errorRoot = document.getElementById('error-root');

    function formatBytes(value) {
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        return '?';
      }
      if (value < 1024) {
        return value + ' B';
      }
      const units = ['KB', 'MB', 'GB', 'TB'];
      let size = value;
      let index = -1;
      while (size >= 1024 && index < units.length - 1) {
        size /= 1024;
        index += 1;
      }
      return size.toFixed(size >= 10 || index === 0 ? 1 : 2) + ' ' + units[index];
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function splitTensorName(name) {
      const parts = String(name).split('.').filter(Boolean);
      return parts.length === 0 ? [String(name)] : parts;
    }

    function normalizeText(value) {
      return String(value || '').toLowerCase();
    }

    function iconMarkup(name) {
      switch (name) {
        case 'chevron':
          return '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M4 2.5L8 6L4 9.5"></path></svg>';
        case 'load':
          return '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.5v6"></path><path d="M5.5 6.5L8 9l2.5-2.5"></path><path d="M3 11.5h10"></path></svg>';
        case 'more':
          return '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.5v11"></path><path d="M3 8h10"></path></svg>';
        case 'offset':
          return '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 4.5h11"></path><path d="M2.5 8h11"></path><path d="M2.5 11.5h11"></path><path d="M5 3v10"></path></svg>';
        default:
          return '';
      }
    }

    function isFiniteDimension(value) {
      return typeof value === 'number' && Number.isFinite(value) && value >= 0;
    }

    function getOffsetMode(tensor) {
      const shape = Array.isArray(tensor.shape) ? tensor.shape : [];
      if (shape.length === 0) {
        return 'scalar';
      }
      return shape.every((dimension) => isFiniteDimension(dimension)) ? 'multi' : 'linear';
    }

    function clampOffset(value, max) {
      const number = Number.isFinite(value) ? Math.trunc(value) : 0;
      if (!Number.isFinite(max)) {
        return Math.max(0, number);
      }
      return Math.max(0, Math.min(number, Math.max(0, max)));
    }

    function getStoredOffsetVector(tensor) {
      if (state.offsets.has(tensor.name)) {
        return state.offsets.get(tensor.name).slice();
      }

      const mode = getOffsetMode(tensor);
      const defaults = mode === 'multi' ? new Array(tensor.shape.length).fill(0) : [0];
      state.offsets.set(tensor.name, defaults);
      return defaults.slice();
    }

    function getLinearOffsetFromVector(tensor, vector) {
      const mode = getOffsetMode(tensor);
      if (mode === 'scalar') {
        return 0;
      }
      if (mode === 'linear') {
        return clampOffset(vector[0], Number.MAX_SAFE_INTEGER);
      }

      const shape = tensor.shape;
      let stride = 1;
      let linear = 0;
      for (let index = shape.length - 1; index >= 0; index -= 1) {
        const dimension = Math.max(1, Math.trunc(shape[index]));
        const value = clampOffset(vector[index], dimension - 1);
        linear += value * stride;
        stride *= dimension;
      }
      return linear;
    }

    function linearOffsetToVector(tensor, linearOffset) {
      const mode = getOffsetMode(tensor);
      if (mode === 'scalar') {
        return [];
      }
      if (mode === 'linear') {
        return [clampOffset(linearOffset, Number.MAX_SAFE_INTEGER)];
      }

      const shape = tensor.shape.map((dimension) => Math.max(1, Math.trunc(dimension)));
      const values = new Array(shape.length).fill(0);
      let remaining = Math.max(0, Math.trunc(linearOffset));
      for (let index = shape.length - 1; index >= 0; index -= 1) {
        const dimension = shape[index];
        values[index] = remaining % dimension;
        remaining = Math.floor(remaining / dimension);
      }
      return values;
    }

    function formatOffsetDisplay(tensor, linearOffset) {
      const mode = getOffsetMode(tensor);
      if (mode === 'scalar') {
        return '[]';
      }
      if (mode === 'linear') {
        return String(Math.max(0, Math.trunc(linearOffset)));
      }
      return '[' + linearOffsetToVector(tensor, linearOffset).join(', ') + ']';
    }

    function getVisibleTensors() {
      if (!state.manifest) {
        return [];
      }

      const text = normalizeText(state.filterText).trim();
      if (!text) {
        return state.manifest.tensors;
      }

      return state.manifest.tensors.filter((item) => {
        const name = normalizeText(item.name);
        return state.prefixOnly ? name.startsWith(text) : name.includes(text);
      });
    }

    function buildTree(items) {
      const root = new Map();
      for (const item of items) {
        const parts = splitTensorName(item.name);
        let cursor = root;
        const prefix = [];
        for (let index = 0; index < parts.length; index += 1) {
          const part = parts[index];
          prefix.push(part);
          if (!cursor.has(part)) {
            cursor.set(part, {
              label: part,
              fullName: prefix.join('.'),
              children: new Map(),
              item: null
            });
          }
          const node = cursor.get(part);
          if (index === parts.length - 1) {
            node.item = item;
          }
          cursor = node.children;
        }
      }
      return root;
    }

    function getDirectChildCount(node) {
      return node.children.size + (node.item ? 1 : 0);
    }

    function renderMetadata() {
      const entries = state.manifest && Array.isArray(state.manifest.metadataEntries) ? state.manifest.metadataEntries : [];
      if (!entries || entries.length === 0) {
        metadataRoot.innerHTML = '';
        return;
      }

      const rows = entries
        .map((entry) => '<dt>' + escapeHtml(entry.key) + '</dt><dd>' + escapeHtml(entry.value) + '</dd>')
        .join('');

      metadataRoot.innerHTML = [
        '<div class="card">',
        '<div class="section-title">Metadata</div>',
        '<dl class="metadata-grid">',
        rows,
        '</dl>',
        '</div>'
      ].join('');
    }

    function renderTree() {
      if (!state.manifest || !Array.isArray(state.manifest.tensors) || state.manifest.tensors.length === 0) {
        treeRoot.className = 'tree-empty';
        treeRoot.textContent = 'No tensor entries were found.';
        return;
      }

      const visibleTensors = getVisibleTensors();
      if (visibleTensors.length === 0) {
        treeRoot.className = 'tree-empty';
        treeRoot.textContent = 'No tensors match the current filter.';
        return;
      }

      treeRoot.className = 'tree';
      treeRoot.innerHTML = '';
      const header = document.createElement('div');
      header.className = 'tree-header';
      header.innerHTML = '<div>Tensors</div><div>Shape</div><div>Precision</div>';
      treeRoot.appendChild(header);

      const body = document.createElement('div');
      body.className = 'tree-body';

      const fragment = document.createDocumentFragment();
      const tree = buildTree(visibleTensors);
      const entries = Array.from(tree.values()).sort((left, right) => left.label.localeCompare(right.label));
      for (const entry of entries) {
        fragment.appendChild(renderTreeNode(entry, 0));
      }
      body.appendChild(fragment);
      treeRoot.appendChild(body);
    }

    function splitNameParts(fullText) {
      const text = String(fullText);
      const index = text.lastIndexOf('.');
      if (index === -1) {
        return {
          prefix: '',
          name: text
        };
      }
      return {
        prefix: text.slice(0, index + 1),
        name: text.slice(index + 1)
      };
    }

    function createTreeRow({ fullText, shapeText = '', dtype = '', groupCount = null }) {
      const parts = splitNameParts(fullText);
      const row = document.createElement('div');
      row.className = 'tree-row';

      const nameCell = document.createElement('div');
      nameCell.className = 'tree-name-cell';
      const text = document.createElement('div');
      text.className = 'tree-name-text';

      if (parts.prefix) {
        const prefix = document.createElement('span');
        prefix.className = 'tree-path-prefix';
        prefix.textContent = parts.prefix;
        text.appendChild(prefix);
      }

      const name = document.createElement('span');
      name.className = 'tree-path-name';
      name.textContent = parts.name;
      text.appendChild(name);

      if (Number.isInteger(groupCount)) {
        const groupMeta = document.createElement('span');
        groupMeta.className = 'tree-group-meta';

        const count = document.createElement('span');
        count.textContent = '(' + String(groupCount) + ')';
        groupMeta.appendChild(count);

        const caret = document.createElement('span');
        caret.className = 'tree-caret';
        caret.innerHTML = iconMarkup('chevron');
        groupMeta.appendChild(caret);

        text.appendChild(groupMeta);
      }

      nameCell.appendChild(text);
      row.appendChild(nameCell);

      const shapeCell = document.createElement('div');
      shapeCell.className = 'tree-shape-cell';
      shapeCell.textContent = shapeText || '';
      row.appendChild(shapeCell);

      const precisionCell = document.createElement('div');
      precisionCell.className = 'tree-precision-cell';
      precisionCell.textContent = dtype || '';
      row.appendChild(precisionCell);

      return row;
    }

    function renderTreeNode(node, depth) {
      if (node.children.size === 0 && node.item) {
        const button = document.createElement('button');
        button.className = 'tensor-leaf';
        if (node.item.name === state.selectedName) {
          button.classList.add('active');
        }
        button.type = 'button';
        button.title = node.item.name;
        button.appendChild(createTreeRow({
          fullText: node.item.name,
          shapeText: node.item.shapeText || '[]',
          dtype: node.item.dtype || '?'
        }));
        button.addEventListener('click', () => {
          selectTensor(node.item.name);
        });
        return button;
      }

      const container = document.createElement('details');
      const shouldOpen = state.filterText ? true : (depth < 1 || state.openGroups.has(node.fullName));
      if (shouldOpen) {
        container.open = true;
      }
      if (node.fullName === state.selectedName) {
        container.classList.add('selected');
      }
      const summary = document.createElement('summary');
      summary.appendChild(createTreeRow({
        fullText: node.fullName,
        groupCount: getDirectChildCount(node)
      }));
      container.appendChild(summary);
      container.addEventListener('toggle', () => {
        if (container.open) {
          state.openGroups.add(node.fullName);
        } else {
          state.openGroups.delete(node.fullName);
        }
      });

      const children = Array.from(node.children.values()).sort((left, right) => left.label.localeCompare(right.label));
      for (const child of children) {
        container.appendChild(renderTreeNode(child, depth + 1));
      }

      if (node.item) {
        container.appendChild(renderTreeNode({
          label: node.item.name,
          fullName: node.item.name,
          children: new Map(),
          item: node.item
        }, depth + 1));
      }

      return container;
    }

    function renderDetail() {
      if (!state.selectedName) {
        detailRoot.className = 'detail-empty';
        detailRoot.textContent = 'Select a tensor from the left tree. Values are loaded on demand from the remote host only when you ask for them.';
        return;
      }

      const details = state.details.get(state.selectedName);
      if (!details) {
        detailRoot.className = 'detail-empty';
        detailRoot.textContent = 'Loading tensor metadata...';
        return;
      }

      const tensor = details.tensor;
      const preview = state.previews.get(state.selectedName) || null;
      const offsetMode = getOffsetMode(tensor);
      const offsetVector = getStoredOffsetVector(tensor);
      const linearOffset = getLinearOffsetFromVector(tensor, offsetVector);
      detailRoot.className = '';

      const previewBlock = (() => {
        if (!preview) {
          return '<div class="hint">No tensor values loaded yet.</div>';
        }
        if (preview.loading) {
          return '<div class="detail-empty">Loading tensor values from the extension host...</div>';
        }
        if (!preview.supported) {
          return '<div class="error">' + escapeHtml(preview.reason) + '</div>';
        }
        const previewMeta = '<div class="preview-meta">Offset ' +
          escapeHtml(formatOffsetDisplay(tensor, preview.startOffset || 0)) +
          ' (linear ' +
          escapeHtml(String(preview.startOffset || 0)) +
          ') • Loaded ' +
          escapeHtml(String(preview.values.length)) +
          ' / ' +
          escapeHtml(String(preview.totalElements)) +
          ' values.</div>';
        return previewMeta + '<pre>' + escapeHtml(JSON.stringify(preview.values, null, 2)) + '</pre>';
      })();

      const canLoadValues = tensor.previewSupported;
      const canLoadMore = Boolean(preview && preview.supported && preview.hasMore && !preview.loading);
      const offsetFields = (() => {
        if (offsetMode === 'scalar') {
          return '<div class="offset-caption">Scalar tensor. Offset is fixed at [] (linear 0).</div>';
        }
        if (offsetMode === 'linear') {
          return [
            '<div class="offset-grid">',
            '<div class="offset-field">',
            '<label for="linear-offset-input">Linear offset</label>',
            '<input id="linear-offset-input" data-offset-input="linear" type="number" min="0" step="1" value="' + escapeHtml(String(offsetVector[0] || 0)) + '" />',
            '</div>',
            '</div>'
          ].join('');
        }
        return [
          '<div class="offset-grid">',
          tensor.shape.map((dimension, index) => (
            '<div class="offset-field">' +
              '<label for="offset-input-' + String(index) + '">Dim ' + String(index) + ' / ' + escapeHtml(String(dimension)) + '</label>' +
              '<input id="offset-input-' + String(index) + '" data-offset-input="' + String(index) + '" type="number" min="0" max="' + escapeHtml(String(Math.max(0, Math.trunc(dimension) - 1))) + '" step="1" value="' + escapeHtml(String(offsetVector[index] || 0)) + '" />' +
            '</div>'
          )).join(''),
          '</div>'
        ].join('');
      })();

      detailRoot.innerHTML = [
        '<div class="tensor-name">' + escapeHtml(tensor.name) + '</div>',
        '<dl class="detail-grid">',
        '<dt>DType</dt><dd>' + escapeHtml(tensor.dtype || '?') + '</dd>',
        '<dt>Shape</dt><dd>' + escapeHtml(tensor.shapeText || '[]') + '</dd>',
        '<dt>Elements</dt><dd>' + escapeHtml(tensor.totalElements || '?') + '</dd>',
        '<dt>Bytes</dt><dd>' + escapeHtml(formatBytes(tensor.byteLength)) + '</dd>',
        '<dt>Shard</dt><dd>' + escapeHtml(tensor.sourceFile || '') + '</dd>',
        '</dl>',
        '<div class="offset-panel">',
        '<div class="section-title"><span class="button-content"><span class="button-icon">' + iconMarkup('offset') + '</span><span>Preview Offset</span></span></div>',
        offsetFields,
        '<div class="offset-caption">Current offset: <span id="offset-display">' + escapeHtml(offsetMode === 'scalar' ? '[]' : formatOffsetDisplay(tensor, linearOffset)) + '</span> <span id="linear-offset-display">(' + (offsetMode === 'scalar' ? 'linear 0' : 'linear ' + escapeHtml(String(linearOffset))) + ')</span></div>',
        '</div>',
        '<div class="actions">',
        '<button class="button" id="load-values-button"' + (canLoadValues && !(preview && preview.loading) ? '' : ' disabled') + '><span class="button-content"><span class="button-icon">' + iconMarkup('load') + '</span><span>' + (preview && preview.supported ? 'Reload From Offset' : 'Load Tensor Values') + '</span></span></button>',
        '<button class="button secondary" id="load-more-button"' + (canLoadMore ? '' : ' disabled') + '><span class="button-content"><span class="button-icon">' + iconMarkup('more') + '</span><span>Load More</span></span></button>',
        '<span class="hint">Only the requested value slice is read from the remote file.</span>',
        '</div>',
        previewBlock
      ].join('');

      const offsetInputs = detailRoot.querySelectorAll('[data-offset-input]');
      for (const input of offsetInputs) {
        input.addEventListener('input', () => {
          updateTensorOffsetInputs(tensor);
        });
        input.addEventListener('change', () => {
          updateTensorOffsetInputs(tensor);
        });
      }

      const loadButton = document.getElementById('load-values-button');
      if (loadButton && canLoadValues) {
        loadButton.addEventListener('click', () => {
          loadTensorValues(state.selectedName, false);
        });
      }

      const loadMoreButton = document.getElementById('load-more-button');
      if (loadMoreButton && canLoadMore) {
        loadMoreButton.addEventListener('click', () => {
          loadTensorValues(state.selectedName, true);
        });
      }
    }

    function updateTensorOffsetInputs(tensor) {
      const mode = getOffsetMode(tensor);
      const next = [];

      if (mode === 'linear') {
        const input = document.getElementById('linear-offset-input');
        const value = input ? Number.parseInt(input.value || '0', 10) : 0;
        next.push(clampOffset(value, Number.MAX_SAFE_INTEGER));
        if (input) {
          input.value = String(next[0]);
        }
      } else if (mode === 'multi') {
        for (let index = 0; index < tensor.shape.length; index += 1) {
          const input = document.getElementById('offset-input-' + String(index));
          const value = input ? Number.parseInt(input.value || '0', 10) : 0;
          const max = Math.max(0, Math.trunc(tensor.shape[index]) - 1);
          const clamped = clampOffset(value, max);
          next.push(clamped);
          if (input) {
            input.value = String(clamped);
          }
        }
      }

      state.offsets.set(tensor.name, next);
      const linear = getLinearOffsetFromVector(tensor, next);
      const offsetDisplay = document.getElementById('offset-display');
      const linearDisplay = document.getElementById('linear-offset-display');
      if (offsetDisplay) {
        offsetDisplay.textContent = mode === 'scalar' ? '[]' : formatOffsetDisplay(tensor, linear);
      }
      if (linearDisplay) {
        linearDisplay.textContent = '(linear ' + String(linear) + ')';
      }
    }

    function renderManifest() {
      title.textContent = state.manifest.title;
      subtitle.textContent =
        state.manifest.mode === 'index'
          ? 'Tensor names, shapes and dtypes come from the index and shard headers. Tensor values stay lazy until requested.'
          : 'Tensor names, shapes and dtypes come from the safetensors header. Tensor values stay lazy until requested.';
      tensorCount.textContent = state.manifest.tensorCount + (state.manifest.tensorCount === 1 ? ' tensor' : ' tensors');
      shardCount.textContent = state.manifest.shardCount + (state.manifest.shardCount === 1 ? ' shard' : ' shards');
      state.tensorIndex = new Map(state.manifest.tensors.map((tensor) => [tensor.name, tensor]));
      state.details = new Map(state.manifest.tensors.map((tensor) => [tensor.name, { tensor }]));
      renderMetadata();
      renderTree();
      renderDetail();
    }

    function selectTensor(name) {
      state.selectedName = name;
      renderTree();
      renderDetail();
    }

    function loadTensorValues(name, append) {
      const tensor = state.tensorIndex.get(name);
      const current = state.previews.get(name);
      const nextOffset = append && current && current.supported ?
        current.nextOffset :
        (tensor ? getLinearOffsetFromVector(tensor, getStoredOffsetVector(tensor)) : 0);
      const startOffset = append && current && current.supported ? current.startOffset : nextOffset;
      state.previews.set(name, {
        supported: current ? current.supported : true,
        reason: current ? current.reason : null,
        dtype: current ? current.dtype : null,
        values: append && current && Array.isArray(current.values) ? current.values.slice() : [],
        totalElements: current ? current.totalElements : '?',
        startOffset,
        nextOffset,
        hasMore: current ? current.hasMore : false,
        loading: true,
        appendPending: append
      });
      renderDetail();
      vscode.postMessage({
        type: 'load-values',
        name,
        offset: nextOffset
      });
    }

    window.addEventListener('message', (event) => {
      const message = event.data;
      switch (message.type) {
        case 'init':
          state.manifest = message.payload;
          renderManifest();
          break;
        case 'tensor-details':
          state.details.set(message.payload.tensor.name, message.payload);
          renderDetail();
          break;
        case 'tensor-preview': {
          const incoming = message.payload.preview;
          const current = state.previews.get(message.payload.name);
          if (!incoming.supported) {
            state.previews.set(message.payload.name, {
              supported: false,
              reason: incoming.reason,
              values: [],
              startOffset: current && Number.isFinite(current.startOffset) ? current.startOffset : 0,
              hasMore: false,
              nextOffset: 0,
              loading: false
            });
          } else {
            const existingValues = current && current.supported && Array.isArray(current.values) && current.appendPending ? current.values : [];
            state.previews.set(message.payload.name, {
              supported: true,
              dtype: incoming.dtype,
              values: existingValues.concat(incoming.values),
              totalElements: incoming.totalElements,
              startOffset: current && Number.isFinite(current.startOffset) ? current.startOffset : incoming.offset,
              nextOffset: incoming.nextOffset,
              hasMore: incoming.hasMore,
              loading: false
            });
          }
          renderDetail();
          break;
        }
        case 'error':
          errorRoot.style.display = 'block';
          errorRoot.className = 'error';
          errorRoot.textContent = message.payload.message;
          break;
        default:
          break;
      }
    });

    searchInput.addEventListener('input', (event) => {
      state.filterText = event.target.value || '';
      renderTree();
    });

    prefixToggle.addEventListener('change', (event) => {
      state.prefixOnly = Boolean(event.target.checked);
      renderTree();
    });

    vscode.postMessage({ type: 'ready' });
  </script>
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

module.exports = {
  registerSafetensorsExplorer,
  _internal: {
    readSafetensorsHeader,
    readTensorPreview,
    decodeFloat16,
    decodeBFloat16,
    getDecoder
  }
};
