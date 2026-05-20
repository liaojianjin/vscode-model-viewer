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
    const records = await this.getAllRecords();
    const tensors = records.map((record) => toTensorSummary(record, records));

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
    const records = await this.getAllRecords();
    return {
      tensor: toTensorSummary(record, records)
    };
  }

  async getTensorPreview(name, options = {}) {
    const record = await this.getRecord(name);
    const preview = await readTensorPreview(record, options);
    preview.dequantized = await this.getDequantizedPreview(record, preview, options);
    return {
      name: record.name,
      preview
    };
  }

  async getDequantizedPreview(record, preview, options = {}) {
    if (!preview || !preview.supported || !Array.isArray(preview.values) || preview.values.length === 0) {
      return null;
    }

    const records = await this.getAllRecords();
    const plan = buildDequantizationPlan(record, records);
    if (!plan) {
      return null;
    }
    if (!plan.supported) {
      return plan;
    }
    return await applyDequantizationPreview(record, preview, plan, options);
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
  const bitsPerElement = decoder.bitsPerElement;
  const maxElementsPerPage = Math.max(1, Math.floor((MAX_PREVIEW_BYTES * 8) / bitsPerElement));

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

  const bitOffset = offset * bitsPerElement;
  const byteOffset = Math.floor(bitOffset / 8);
  const bitShift = bitOffset % 8;
  const availableBitsFromBytes = Math.max(0, (record.byteLength * 8) - bitOffset);
  const availableElementsFromBytes = Math.max(0, Math.floor(availableBitsFromBytes / bitsPerElement));
  const remainingElements = totalElementNumber === null ? availableElementsFromBytes : Math.max(0, totalElementNumber - offset);
  const requestedElements = Math.max(0, Math.min(remainingElements, availableElementsFromBytes, limit, maxElementsPerPage));
  const byteLength = requestedElements > 0 ? Math.ceil((bitShift + (requestedElements * bitsPerElement)) / 8) : 0;
  const buffer = byteLength > 0 ? await readFileRange(record.filePath, record.dataOffset + byteOffset, byteLength) : Buffer.alloc(0);
  const values = decodeValues(buffer, decoder, requestedElements, bitShift);
  const nextOffset = offset + values.length;
  const hasMore = totalElementNumber === null ?
    ((record.byteLength * 8) > nextOffset * bitsPerElement) :
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

function buildDequantizationPlan(record, records) {
  const decoder = getDecoder(record.dtype);
  const dtype = normalizeDtype(record.dtype);
  const scaleMatch = findRelatedTensor(record.name, records, getScaleCandidateNames(record.name));
  const zeroPointMatch = findRelatedTensor(record.name, records, getZeroPointCandidateNames(record.name));
  const isDirectQuantizedValue = decoder && (decoder.quantized || dtype === 'I8' || dtype === 'U8');

  if (!scaleMatch) {
    if (!decoder || !decoder.quantized) {
      return null;
    }
    return {
      supported: false,
      reason: 'No related scale tensor was found. The stored dtype values are shown instead.'
    };
  }

  if (!isDirectQuantizedValue) {
    return {
      supported: false,
      reason: `A related scale tensor was found, but dequantization for packed or non-direct dtype "${record.dtype}" is not implemented.`
    };
  }

  if (!getDecoder(scaleMatch.record.dtype)) {
    return {
      supported: false,
      reason: `Scale tensor "${scaleMatch.record.name}" uses unsupported dtype "${scaleMatch.record.dtype}".`
    };
  }

  const scaleMapping = createAuxiliaryTensorMapping(record, scaleMatch.record);
  if (!scaleMapping) {
    return {
      supported: false,
      reason: `Scale tensor "${scaleMatch.record.name}" shape ${formatShape(scaleMatch.record.shape)} cannot be mapped to weight shape ${formatShape(record.shape)}.`
    };
  }

  let zeroPoint = null;
  if (zeroPointMatch) {
    if (!getDecoder(zeroPointMatch.record.dtype)) {
      return {
        supported: false,
        reason: `Zero-point tensor "${zeroPointMatch.record.name}" uses unsupported dtype "${zeroPointMatch.record.dtype}".`
      };
    }
    const zeroPointMapping = createAuxiliaryTensorMapping(record, zeroPointMatch.record);
    if (!zeroPointMapping) {
      return {
        supported: false,
        reason: `Zero-point tensor "${zeroPointMatch.record.name}" shape ${formatShape(zeroPointMatch.record.shape)} cannot be mapped to weight shape ${formatShape(record.shape)}.`
      };
    }
    zeroPoint = {
      record: zeroPointMatch.record,
      mapping: zeroPointMapping
    };
  }

  const scaleOperation = scaleMatch.kind === 'inverse' ? 'divide' : 'multiply';
  return {
    supported: true,
    scaleOperation,
    scale: {
      record: scaleMatch.record,
      mapping: scaleMapping
    },
    zeroPoint
  };
}

async function applyDequantizationPreview(record, preview, plan) {
  const valueCache = new Map();
  const startOffset = Number.isFinite(preview.offset) ? Math.max(0, Math.trunc(preview.offset)) : 0;
  const values = [];

  const readAuxiliaryValue = async (auxRecord, offset) => {
    const key = `${auxRecord.name}:${offset}`;
    if (!valueCache.has(key)) {
      valueCache.set(key, readTensorPreview(auxRecord, { offset, limit: 1 }));
    }
    const auxPreview = await valueCache.get(key);
    if (!auxPreview.supported || !Array.isArray(auxPreview.values) || auxPreview.values.length === 0) {
      return null;
    }
    return toFiniteNumber(auxPreview.values[0]);
  };

  for (let index = 0; index < preview.values.length; index += 1) {
    const tensorOffset = startOffset + index;
    const storedValue = toFiniteNumber(preview.values[index]);
    const scaleOffset = plan.scale.mapping.toAuxiliaryOffset(tensorOffset);
    const scale = await readAuxiliaryValue(plan.scale.record, scaleOffset);

    if (storedValue === null || scale === null || (plan.scaleOperation === 'divide' && scale === 0)) {
      values.push(null);
      continue;
    }

    let centeredValue = storedValue;
    if (plan.zeroPoint) {
      const zeroPointOffset = plan.zeroPoint.mapping.toAuxiliaryOffset(tensorOffset);
      const zeroPoint = await readAuxiliaryValue(plan.zeroPoint.record, zeroPointOffset);
      if (zeroPoint === null) {
        values.push(null);
        continue;
      }
      centeredValue -= zeroPoint;
    }

    values.push(plan.scaleOperation === 'divide' ? centeredValue / scale : centeredValue * scale);
  }

  const zeroPointText = plan.zeroPoint ? ` and zero point "${plan.zeroPoint.record.name}"` : '';
  const operationText = plan.scaleOperation === 'divide' ? '(stored - zeroPoint) / scale_inv' : '(stored - zeroPoint) * scale';
  return {
    supported: true,
    values,
    method: `${operationText} using "${plan.scale.record.name}"${zeroPointText}`,
    scaleTensor: plan.scale.record.name,
    scaleMapping: plan.scale.mapping.description,
    zeroPointTensor: plan.zeroPoint ? plan.zeroPoint.record.name : null
  };
}

function findRelatedTensor(name, records, candidateNames) {
  const byName = new Map(records.map((record) => [record.name, record]));
  for (const candidate of candidateNames) {
    if (byName.has(candidate.name)) {
      return {
        record: byName.get(candidate.name),
        kind: candidate.kind
      };
    }
  }
  return null;
}

function getScaleCandidateNames(name) {
  const candidates = [];
  const add = (candidateName, kind = 'scale') => {
    if (candidateName && candidateName !== name && !candidates.some((candidate) => candidate.name === candidateName)) {
      candidates.push({ name: candidateName, kind });
    }
  };

  add(`${name}_scale`);
  add(`${name}_scales`);
  add(`${name}.scale`);
  add(`${name}.scales`);
  add(`${name}_scale_inv`, 'inverse');
  add(`${name}.scale_inv`, 'inverse');

  for (const suffix of ['.weight', '_weight']) {
    if (name.endsWith(suffix)) {
      const base = name.slice(0, -suffix.length);
      add(`${base}${suffix}_scale`);
      add(`${base}${suffix}_scales`);
      add(`${base}${suffix}.scale`);
      add(`${base}${suffix}.scales`);
      add(`${base}${suffix}_scale_inv`, 'inverse');
      add(`${base}${suffix}.scale_inv`, 'inverse');
      add(`${base}.scale`);
      add(`${base}.scales`);
      add(`${base}_scale`);
      add(`${base}_scales`);
      add(`${base}.weight_scale`);
      add(`${base}.weight_scales`);
      add(`${base}.weight_scale_inv`, 'inverse');
      add(`${base}_weight_scale`);
      add(`${base}_weight_scales`);
      add(`${base}_weight_scale_inv`, 'inverse');
    }
  }

  return candidates;
}

function getZeroPointCandidateNames(name) {
  const candidates = [];
  const add = (candidateName) => {
    if (candidateName && candidateName !== name && !candidates.some((candidate) => candidate.name === candidateName)) {
      candidates.push({ name: candidateName, kind: 'zeroPoint' });
    }
  };

  add(`${name}_zero`);
  add(`${name}_zeros`);
  add(`${name}_zero_point`);
  add(`${name}.zero`);
  add(`${name}.zeros`);
  add(`${name}.zero_point`);

  for (const suffix of ['.weight', '_weight']) {
    if (name.endsWith(suffix)) {
      const base = name.slice(0, -suffix.length);
      add(`${base}${suffix}_zero`);
      add(`${base}${suffix}_zeros`);
      add(`${base}${suffix}_zero_point`);
      add(`${base}${suffix}.zero`);
      add(`${base}${suffix}.zeros`);
      add(`${base}${suffix}.zero_point`);
      add(`${base}.zero`);
      add(`${base}.zeros`);
      add(`${base}.zero_point`);
      add(`${base}_zero`);
      add(`${base}_zeros`);
      add(`${base}_zero_point`);
      add(`${base}.weight_zero`);
      add(`${base}.weight_zeros`);
      add(`${base}.weight_zero_point`);
      add(`${base}_weight_zero`);
      add(`${base}_weight_zeros`);
      add(`${base}_weight_zero_point`);
    }
  }

  return candidates;
}

function createAuxiliaryTensorMapping(weightRecord, auxiliaryRecord) {
  const weightShape = toValidShape(weightRecord.shape);
  const auxiliaryShape = toValidShape(auxiliaryRecord.shape);
  const auxiliaryElements = getSafeElementCountNumber(auxiliaryRecord.elementCount);
  const weightElements = getSafeElementCountNumber(weightRecord.elementCount);

  if (!auxiliaryShape || auxiliaryElements === null || auxiliaryElements <= 0) {
    return null;
  }

  if (auxiliaryElements === 1) {
    return {
      description: 'scalar',
      toAuxiliaryOffset: () => 0
    };
  }

  if (weightElements !== null && auxiliaryElements === weightElements) {
    return {
      description: 'per-element',
      toAuxiliaryOffset: (linearOffset) => Math.max(0, Math.min(auxiliaryElements - 1, Math.trunc(linearOffset)))
    };
  }

  if (weightShape && auxiliaryShape && weightShape.length === auxiliaryShape.length && auxiliaryShape.every((dimension, index) => dimension > 0 && dimension <= weightShape[index])) {
    const weightStrides = getStrides(weightShape);
    const auxiliaryStrides = getStrides(auxiliaryShape);
    const blockSizes = weightShape.map((dimension, index) => Math.max(1, Math.ceil(dimension / auxiliaryShape[index])));
    return {
      description: `per-block grid ${formatShape(auxiliaryShape)}, block ${formatShape(blockSizes)}`,
      toAuxiliaryOffset: (linearOffset) => {
        const indices = linearOffsetToIndices(weightShape, weightStrides, linearOffset);
        return indices.reduce((sum, value, index) => {
          const auxiliaryIndex = Math.min(auxiliaryShape[index] - 1, Math.floor(value / blockSizes[index]));
          return sum + (auxiliaryIndex * auxiliaryStrides[index]);
        }, 0);
      }
    };
  }

  if (weightShape && auxiliaryShape && auxiliaryShape.length === 1) {
    const auxiliaryLength = auxiliaryShape[0];
    const weightStrides = getStrides(weightShape);
    if (auxiliaryLength === weightShape[0]) {
      return {
        description: 'per first dimension',
        toAuxiliaryOffset: (linearOffset) => linearOffsetToIndices(weightShape, weightStrides, linearOffset)[0]
      };
    }
    const lastIndex = weightShape.length - 1;
    if (auxiliaryLength === weightShape[lastIndex]) {
      return {
        description: 'per last dimension',
        toAuxiliaryOffset: (linearOffset) => linearOffsetToIndices(weightShape, weightStrides, linearOffset)[lastIndex]
      };
    }
  }

  if (weightElements !== null && auxiliaryElements < weightElements) {
    const blockSize = Math.max(1, Math.ceil(weightElements / auxiliaryElements));
    return {
      description: `linear blocks of ${blockSize}`,
      toAuxiliaryOffset: (linearOffset) => Math.max(0, Math.min(auxiliaryElements - 1, Math.floor(Math.trunc(linearOffset) / blockSize)))
    };
  }

  return null;
}

function toValidShape(shape) {
  if (!Array.isArray(shape)) {
    return null;
  }
  const values = shape.map((dimension) => Number(dimension));
  return values.every((dimension) => Number.isFinite(dimension) && dimension >= 0) ? values.map((dimension) => Math.trunc(dimension)) : null;
}

function getSafeElementCountNumber(value) {
  return value !== null && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null;
}

function getStrides(shape) {
  const strides = new Array(shape.length).fill(1);
  for (let index = shape.length - 2; index >= 0; index -= 1) {
    strides[index] = strides[index + 1] * Math.max(1, shape[index + 1]);
  }
  return strides;
}

function linearOffsetToIndices(shape, strides, linearOffset) {
  const indices = new Array(shape.length).fill(0);
  let remaining = Math.max(0, Math.trunc(linearOffset));
  for (let index = 0; index < shape.length; index += 1) {
    const stride = strides[index];
    indices[index] = Math.min(Math.max(0, shape[index] - 1), Math.floor(remaining / stride));
    remaining %= stride;
  }
  return indices;
}

function toFiniteNumber(value) {
  const number = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function decodeValues(buffer, decoder, requestedElements = Number.MAX_SAFE_INTEGER, startBitOffset = 0) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const values = [];

  if (decoder.bitsPerElement % 8 === 0) {
    const bytesPerElement = decoder.bitsPerElement / 8;
    for (let offset = 0; offset + bytesPerElement <= buffer.byteLength && values.length < requestedElements; offset += bytesPerElement) {
      values.push(decoder.read(view, offset));
    }
    return values;
  }

  for (let index = 0; index < requestedElements; index += 1) {
    const bitOffset = startBitOffset + (index * decoder.bitsPerElement);
    if (bitOffset + decoder.bitsPerElement > buffer.byteLength * 8) {
      break;
    }
    values.push(decoder.readBits(readUnsignedBits(buffer, bitOffset, decoder.bitsPerElement)));
  }
  return values;
}

function readUnsignedBits(buffer, bitOffset, bitLength) {
  let value = 0;
  for (let index = 0; index < bitLength; index += 1) {
    const absoluteBit = bitOffset + index;
    const byte = buffer[Math.floor(absoluteBit / 8)];
    const bit = (byte >> (absoluteBit % 8)) & 1;
    value |= bit << index;
  }
  return value;
}

function getDecoder(dtype) {
  const normalizedDtype = normalizeDtype(dtype);
  const decoders = {
    BOOL: { bitsPerElement: 8, read: (view, offset) => view.getUint8(offset) !== 0 },
    F4: { bitsPerElement: 4, quantized: true, description: 'MXFP4 decoded from packed 4-bit values.', readBits: decodeFloat4E2M1 },
    F6_E2M3: { bitsPerElement: 6, quantized: true, description: 'MXFP6 E2M3 decoded from packed 6-bit values.', readBits: (value) => decodePackedFloat(value, 2, 3, 1, false) },
    F6_E3M2: { bitsPerElement: 6, quantized: true, description: 'MXFP6 E3M2 decoded from packed 6-bit values.', readBits: (value) => decodePackedFloat(value, 3, 2, 3, false) },
    U8: { bitsPerElement: 8, read: (view, offset) => view.getUint8(offset) },
    I8: { bitsPerElement: 8, read: (view, offset) => view.getInt8(offset) },
    F8_E5M2: { bitsPerElement: 8, quantized: true, description: 'FP8 E5M2 decoded to approximate JavaScript numbers.', read: (view, offset) => decodePackedFloat(view.getUint8(offset), 5, 2, 15, true) },
    F8_E4M3: { bitsPerElement: 8, quantized: true, description: 'FP8 E4M3 decoded to approximate JavaScript numbers.', read: (view, offset) => decodeFloat8E4M3(view.getUint8(offset)) },
    F8_E8M0: { bitsPerElement: 8, quantized: true, description: 'E8M0 scale values decoded as powers of two.', read: (view, offset) => decodeFloat8E8M0(view.getUint8(offset)) },
    U16: { bitsPerElement: 16, read: (view, offset) => view.getUint16(offset, true) },
    I16: { bitsPerElement: 16, read: (view, offset) => view.getInt16(offset, true) },
    U32: { bitsPerElement: 32, read: (view, offset) => view.getUint32(offset, true) },
    I32: { bitsPerElement: 32, read: (view, offset) => view.getInt32(offset, true) },
    U64: { bitsPerElement: 64, read: (view, offset) => view.getBigUint64(offset, true).toString() },
    I64: { bitsPerElement: 64, read: (view, offset) => view.getBigInt64(offset, true).toString() },
    F64: { bitsPerElement: 64, read: (view, offset) => view.getFloat64(offset, true) },
    F32: { bitsPerElement: 32, read: (view, offset) => view.getFloat32(offset, true) },
    C64: { bitsPerElement: 64, read: (view, offset) => ({ real: view.getFloat32(offset, true), imag: view.getFloat32(offset + 4, true) }) },
    F16: { bitsPerElement: 16, read: (view, offset) => decodeFloat16(view.getUint16(offset, true)) },
    BF16: { bitsPerElement: 16, read: (view, offset) => decodeBFloat16(view.getUint16(offset, true)) }
  };
  return decoders[normalizedDtype] || null;
}

function normalizeDtype(dtype) {
  const value = String(dtype || '').toUpperCase();
  const aliases = {
    F8_E4M3FN: 'F8_E4M3',
    F8_E4M3FNUZ: 'F8_E4M3',
    F8_E5M2FNUZ: 'F8_E5M2',
    FLOAT8_E4M3FN: 'F8_E4M3',
    FLOAT8_E5M2: 'F8_E5M2'
  };
  return aliases[value] || value;
}

function isQuantizedDtype(dtype) {
  const decoder = getDecoder(dtype);
  return Boolean(decoder && decoder.quantized);
}

function getDtypeBits(dtype) {
  const decoder = getDecoder(dtype);
  return decoder ? decoder.bitsPerElement : null;
}

function getDtypeDescription(dtype) {
  const decoder = getDecoder(dtype);
  return decoder && decoder.description ? decoder.description : '';
}

function decodeFloat4E2M1(value) {
  return decodePackedFloat(value, 2, 1, 1, false);
}

function decodeFloat8E4M3(value) {
  const sign = (value & 0x80) ? -1 : 1;
  const exponent = (value >> 3) & 0x0f;
  const mantissa = value & 0x07;

  if (exponent === 0 && mantissa === 0) {
    return sign === 1 ? 0 : -0;
  }
  if (exponent === 0) {
    return sign * (mantissa / 8) * Math.pow(2, -6);
  }
  if (exponent === 0x0f && mantissa === 0x07) {
    return NaN;
  }
  return sign * (1 + (mantissa / 8)) * Math.pow(2, exponent - 7);
}

function decodeFloat8E8M0(value) {
  return value === 0xff ? NaN : Math.pow(2, value - 127);
}

function decodePackedFloat(value, exponentBits, mantissaBits, bias, hasInfinity) {
  const signShift = exponentBits + mantissaBits;
  const sign = (value & (1 << signShift)) ? -1 : 1;
  const exponentMask = (1 << exponentBits) - 1;
  const mantissaMask = (1 << mantissaBits) - 1;
  const exponent = (value >> mantissaBits) & exponentMask;
  const mantissa = value & mantissaMask;

  if (exponent === 0 && mantissa === 0) {
    return sign === 1 ? 0 : -0;
  }

  if (exponent === 0) {
    return sign * (mantissa / Math.pow(2, mantissaBits)) * Math.pow(2, 1 - bias);
  }

  if (hasInfinity && exponent === exponentMask) {
    return mantissa === 0 ? sign * Infinity : NaN;
  }

  return sign * (1 + (mantissa / Math.pow(2, mantissaBits))) * Math.pow(2, exponent - bias);
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

function toTensorSummary(record, records = null) {
  const bitsPerElement = getDtypeBits(record.dtype);
  const dequantizationPlan = Array.isArray(records) ? buildDequantizationPlan(record, records) : null;
  const dequantizationSupported = Boolean(dequantizationPlan && dequantizationPlan.supported);
  const scaleTensorName = dequantizationSupported ? dequantizationPlan.scale.record.name : null;
  const scaleMapping = dequantizationSupported ? dequantizationPlan.scale.mapping.description : null;
  const dtypeDescription = dequantizationSupported
    ? `Dequantized preview will use "${scaleTensorName}" (${scaleMapping}).`
    : getDtypeDescription(record.dtype);
  return {
    name: record.name,
    dtype: record.dtype,
    shape: record.shape,
    shapeText: formatShape(record.shape),
    sourceFile: record.sourceFile,
    byteLength: record.byteLength,
    bitsPerElement,
    quantized: isQuantizedDtype(record.dtype) || dequantizationSupported,
    dtypeDescription,
    dequantizationSupported,
    dequantizationScaleTensor: scaleTensorName,
    dequantizationScaleMapping: scaleMapping,
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
      grid-template-rows: minmax(120px, 3fr) 8px minmax(160px, 2fr);
      background: var(--vscode-panel-border);
    }

    .layout.resizing {
      user-select: none;
      cursor: row-resize;
    }

    .pane {
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      background: var(--vscode-editor-background);
    }

    .pane-resizer {
      width: 100%;
      min-height: 8px;
      padding: 0;
      border: 0;
      background: var(--vscode-panel-border);
      cursor: row-resize;
      position: relative;
    }

    .pane-resizer::before {
      content: "";
      position: absolute;
      left: 50%;
      top: 50%;
      width: 44px;
      height: 2px;
      border-radius: 999px;
      transform: translate(-50%, -50%);
      background: var(--vscode-descriptionForeground);
      opacity: 0.65;
    }

    .pane-resizer:hover::before,
    .pane-resizer:focus-visible::before {
      opacity: 1;
      background: var(--vscode-focusBorder);
    }

    .pane-resizer:focus-visible {
      outline: 1px solid var(--vscode-focusBorder);
      outline-offset: -1px;
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

    .value-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: stretch;
      margin: 0;
      padding: 10px;
      border: 1px solid var(--vscode-panel-border);
      border-radius: 8px;
      background: rgba(18, 20, 24, 0.7);
      overflow: auto;
      min-height: 42px;
      max-height: 180px;
    }

    .value-chip {
      flex: 0 0 auto;
      min-width: 42px;
      max-width: 220px;
      padding: 6px 9px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.035);
      color: var(--vscode-editor-foreground);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.35;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      text-align: right;
    }

    .value-empty {
      color: var(--vscode-descriptionForeground);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }

    .preview-meta,
    .preview-title {
      margin-bottom: 10px;
      color: var(--vscode-descriptionForeground);
    }

    .preview-title {
      margin-top: 12px;
      font-weight: 600;
      color: var(--vscode-foreground);
    }

    .preview-title:first-child {
      margin-top: 0;
    }

    .preview-note {
      margin: 0 0 10px;
      color: var(--vscode-descriptionForeground);
      line-height: 1.45;
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
        grid-template-rows: minmax(120px, 1fr) 8px minmax(220px, auto);
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
    <button id="pane-resizer" class="pane-resizer" type="button" aria-label="Resize panes" title="Drag to resize panes"></button>
    <section class="pane">
      <div class="pane-scroll">
        <div id="detail-root" class="detail-empty">Select a tensor from the left tree. Values are loaded on demand from the remote host only when you ask for them.</div>
        <div id="error-root" style="display:none;"></div>
      </div>
    </section>
  </div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
    const state = {
      manifest: null,
      selectedName: null,
      details: new Map(),
      previews: new Map(),
      tensorIndex: new Map(),
      offsets: new Map(),
      filterText: '',
      prefixOnly: false,
      openGroups: new Set(),
      paneSplitRatio: 0.6
    };

    const title = document.getElementById('title');
    const subtitle = document.getElementById('subtitle');
    const tensorCount = document.getElementById('tensor-count');
    const shardCount = document.getElementById('shard-count');
    const metadataRoot = document.getElementById('metadata-root');
    const layoutRoot = document.querySelector('.layout');
    const paneResizer = document.getElementById('pane-resizer');
    const searchInput = document.getElementById('search-input');
    const prefixToggle = document.getElementById('prefix-toggle');
    const treeRoot = document.getElementById('tree-root');
    const detailRoot = document.getElementById('detail-root');
    const errorRoot = document.getElementById('error-root');

    function clampNumber(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function applyPaneSplit(ratio) {
      if (!layoutRoot) {
        return;
      }
      const resizerHeight = paneResizer ? paneResizer.offsetHeight : 8;
      const available = Math.max(1, layoutRoot.clientHeight - resizerHeight);
      const minTop = Math.min(180, Math.max(96, available * 0.25));
      const minBottom = Math.min(220, Math.max(120, available * 0.25));
      const top = clampNumber(Math.round(available * ratio), minTop, Math.max(minTop, available - minBottom));
      const bottom = Math.max(minBottom, available - top);
      state.paneSplitRatio = top / available;
      layoutRoot.style.gridTemplateRows = top + 'px ' + resizerHeight + 'px ' + bottom + 'px';
    }

    function setPaneSplitFromClientY(clientY) {
      if (!layoutRoot) {
        return;
      }
      const rect = layoutRoot.getBoundingClientRect();
      const resizerHeight = paneResizer ? paneResizer.offsetHeight : 8;
      const available = Math.max(1, rect.height - resizerHeight);
      applyPaneSplit((clientY - rect.top) / available);
    }

    function setupPaneResizer() {
      if (!layoutRoot || !paneResizer) {
        return;
      }

      paneResizer.addEventListener('pointerdown', (event) => {
        event.preventDefault();
        paneResizer.setPointerCapture(event.pointerId);
        layoutRoot.classList.add('resizing');
        setPaneSplitFromClientY(event.clientY);
      });

      paneResizer.addEventListener('pointermove', (event) => {
        if (!paneResizer.hasPointerCapture(event.pointerId)) {
          return;
        }
        setPaneSplitFromClientY(event.clientY);
      });

      const stopResize = (event) => {
        if (paneResizer.hasPointerCapture(event.pointerId)) {
          paneResizer.releasePointerCapture(event.pointerId);
        }
        layoutRoot.classList.remove('resizing');
      };
      paneResizer.addEventListener('pointerup', stopResize);
      paneResizer.addEventListener('pointercancel', stopResize);

      paneResizer.addEventListener('keydown', (event) => {
        if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') {
          return;
        }
        event.preventDefault();
        const delta = event.key === 'ArrowUp' ? -0.04 : 0.04;
        applyPaneSplit(state.paneSplitRatio + delta);
      });

      window.addEventListener('resize', () => {
        applyPaneSplit(state.paneSplitRatio);
      });

      applyPaneSplit(state.paneSplitRatio);
    }

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

    function formatDtypeLabel(tensor) {
      const dtype = tensor && tensor.dtype ? String(tensor.dtype) : '?';
      const bits = tensor && Number.isFinite(tensor.bitsPerElement) ? String(tensor.bitsPerElement) + '-bit' : '';
      const suffix = tensor && tensor.quantized ? ' · q' : '';
      return bits ? dtype + ' · ' + bits + suffix : dtype + suffix;
    }

    function formatPreviewValue(value) {
      if (typeof value === 'number') {
        if (Number.isNaN(value)) {
          return 'NaN';
        }
        if (!Number.isFinite(value)) {
          return String(value);
        }
        return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(8)));
      }
      if (typeof value === 'boolean') {
        return value ? 'true' : 'false';
      }
      if (value === null || value === undefined) {
        return 'null';
      }
      if (typeof value === 'object') {
        return JSON.stringify(value);
      }
      return String(value);
    }

    function renderValueStrip(titleText, values, noteText = '') {
      const items = Array.isArray(values) ? values : [];
      const chips = items.length === 0
        ? '<span class="value-empty">[]</span>'
        : items.map((value, index) => {
          const text = formatPreviewValue(value);
          return '<span class="value-chip" title="#' + String(index) + ' ' + escapeHtml(text) + '">' + escapeHtml(text) + '</span>';
        }).join('');
      const note = noteText ? '<div class="preview-note">' + escapeHtml(noteText) + '</div>' : '';
      return [
        '<div class="preview-title">' + escapeHtml(titleText) + '</div>',
        note,
        '<div class="value-strip" role="list">',
        chips,
        '</div>'
      ].join('');
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

    function compareTreeSegment(leftSegment, rightSegment) {
      return collator.compare(String(leftSegment), String(rightSegment));
    }

    function compareTreeNames(leftName, rightName) {
      const leftParts = splitTensorName(leftName);
      const rightParts = splitTensorName(rightName);
      const length = Math.max(leftParts.length, rightParts.length);
      for (let index = 0; index < length; index += 1) {
        if (index >= leftParts.length) {
          return -1;
        }
        if (index >= rightParts.length) {
          return 1;
        }
        const difference = compareTreeSegment(leftParts[index], rightParts[index]);
        if (difference !== 0) {
          return difference;
        }
      }
      return 0;
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
      const sortedVisibleTensors = visibleTensors.slice().sort((left, right) => compareTreeNames(left.name, right.name));
      const tree = buildTree(sortedVisibleTensors);
      const entries = Array.from(tree.values());
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
          dtype: formatDtypeLabel(node.item)
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

      const children = Array.from(node.children.values());
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

        const dequantized = preview.dequantized || null;
        if (dequantized && dequantized.supported) {
          const method = [
            'Method: ' + dequantized.method,
            dequantized.scaleMapping ? 'scale mapping: ' + dequantized.scaleMapping : ''
          ].filter(Boolean).join('; ');
          return [
            previewMeta,
            renderValueStrip('Dequantized Values', dequantized.values, method),
            renderValueStrip('Stored Values', preview.values)
          ].join('');
        }

        const dequantizedNote = dequantized && !dequantized.supported
          ? '<div class="preview-note">Dequantized preview unavailable: ' + escapeHtml(dequantized.reason) + '</div>'
          : '';
        return previewMeta + dequantizedNote + renderValueStrip(tensor.quantized || dequantizedNote ? 'Stored Values' : 'Values', preview.values);
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
        '<dt>DType</dt><dd>' + escapeHtml(formatDtypeLabel(tensor)) + '</dd>',
        tensor.quantized ? '<dt>Quantized</dt><dd>' + escapeHtml(tensor.dtypeDescription || 'Quantized dtype preview is supported.') + '</dd>' : '',
        tensor.dequantizationSupported ? '<dt>Dequantization</dt><dd>' + escapeHtml('Automatic on Load Tensor Values using ' + tensor.dequantizationScaleTensor + ' (' + tensor.dequantizationScaleMapping + ').') + '</dd>' : '',
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
        dequantized: append && current && current.dequantized ? {
          ...current.dequantized,
          values: Array.isArray(current.dequantized.values) ? current.dequantized.values.slice() : []
        } : (current ? current.dequantized : null),
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
            const existingDequantizedValues = current &&
              current.supported &&
              current.appendPending &&
              current.dequantized &&
              current.dequantized.supported &&
              Array.isArray(current.dequantized.values)
                ? current.dequantized.values
                : [];
            const nextDequantized = incoming.dequantized && incoming.dequantized.supported
              ? {
                ...incoming.dequantized,
                values: existingDequantizedValues.concat(incoming.dequantized.values)
              }
              : incoming.dequantized;
            state.previews.set(message.payload.name, {
              supported: true,
              dtype: incoming.dtype,
              values: existingValues.concat(incoming.values),
              dequantized: nextDequantized,
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

    setupPaneResizer();
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
    buildDequantizationPlan,
    applyDequantizationPreview,
    decodeFloat16,
    decodeBFloat16,
    getDecoder
  }
};
