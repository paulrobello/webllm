import {
	GGUF_DEFAULT_ALIGNMENT,
	GGUF_MAGIC,
	GGUF_VERSION,
	type GgufContext,
	type GgufHeader,
	type GgufKv,
	type GgufTensorInfo,
	GgufValueType,
} from "./gguf-types.js";

// biome-ignore lint/complexity/noStaticOnlyClass: instance methods planned for Phase 2
export class GgufParser {
	static parse(buffer: ArrayBuffer): GgufContext {
		const view = new DataView(buffer);
		let offset = 0;

		const header = readHeader(view, offset);
		offset = 24;

		if (header.magic !== GGUF_MAGIC) {
			throw new Error(
				`Invalid GGUF magic: expected 0x${GGUF_MAGIC.toString(16)}, got 0x${header.magic.toString(16)}`,
			);
		}
		if (header.version !== GGUF_VERSION) {
			throw new Error(
				`Unsupported GGUF version: ${header.version}, expected ${GGUF_VERSION}`,
			);
		}

		const metadata = new Map<string, GgufKv>();
		for (let i = 0; i < header.metadataKvCount; i++) {
			const { kv, newOffset } = readKv(view, offset);
			metadata.set(kv.key, kv);
			offset = newOffset;
		}

		const tensors: GgufTensorInfo[] = [];
		for (let i = 0; i < header.tensorCount; i++) {
			const { info, newOffset } = readTensorInfo(view, offset);
			tensors.push(info);
			offset = newOffset;
		}

		const alignment = getAlignment(metadata);
		const dataOffset = alignTo(offset, alignment);
		const totalDataSize = calculateTotalDataSize(tensors);

		return { header, metadata, tensors, alignment, dataOffset, totalDataSize };
	}

	static getMetadataString(ctx: GgufContext, key: string): string | undefined {
		const kv = ctx.metadata.get(key);
		if (!kv || kv.type !== GgufValueType.STRING) return undefined;
		return kv.value as string;
	}

	static getMetadataNumber(ctx: GgufContext, key: string): number | undefined {
		const kv = ctx.metadata.get(key);
		if (!kv) return undefined;
		if (typeof kv.value === "number") return kv.value;
		return undefined;
	}
}

function readHeader(view: DataView, offset: number): GgufHeader {
	const magic = view.getUint32(offset, true);
	const version = view.getUint32(offset + 4, true);
	const tensorCount = Number(view.getBigUint64(offset + 8, true));
	const metadataKvCount = Number(view.getBigUint64(offset + 16, true));
	return { magic, version, tensorCount, metadataKvCount };
}

function readString(
	view: DataView,
	offset: number,
): { value: string; newOffset: number } {
	const length = Number(view.getBigUint64(offset, true));
	offset += 8;
	const bytes = new Uint8Array(view.buffer, view.byteOffset + offset, length);
	const value = new TextDecoder().decode(bytes);
	return { value, newOffset: offset + length };
}

function readKv(
	view: DataView,
	offset: number,
): { kv: GgufKv; newOffset: number } {
	const { value: key, newOffset: afterKey } = readString(view, offset);
	offset = afterKey;
	const type = view.getUint32(offset, true) as GgufValueType;
	offset += 4;
	const { value, newOffset } = readValue(view, offset, type, false);
	return { kv: { key, type, isArray: false, value }, newOffset };
}

function readValue(
	view: DataView,
	offset: number,
	type: GgufValueType,
	_isArray: boolean,
): { value: unknown; newOffset: number } {
	switch (type) {
		case GgufValueType.UINT8:
			return { value: view.getUint8(offset), newOffset: offset + 1 };
		case GgufValueType.INT8:
			return { value: view.getInt8(offset), newOffset: offset + 1 };
		case GgufValueType.UINT16:
			return { value: view.getUint16(offset, true), newOffset: offset + 2 };
		case GgufValueType.INT16:
			return { value: view.getInt16(offset, true), newOffset: offset + 2 };
		case GgufValueType.UINT32:
			return { value: view.getUint32(offset, true), newOffset: offset + 4 };
		case GgufValueType.INT32:
			return { value: view.getInt32(offset, true), newOffset: offset + 4 };
		case GgufValueType.FLOAT32:
			return { value: view.getFloat32(offset, true), newOffset: offset + 4 };
		case GgufValueType.BOOL:
			return { value: view.getUint8(offset) !== 0, newOffset: offset + 1 };
		case GgufValueType.STRING:
			return readString(view, offset);
		case GgufValueType.ARRAY: {
			const elemType = view.getUint32(offset, true) as GgufValueType;
			offset += 4;
			const count = Number(view.getBigUint64(offset, true));
			offset += 8;
			const arr: unknown[] = [];
			for (let i = 0; i < count; i++) {
				const { value, newOffset } = readValue(view, offset, elemType, true);
				arr.push(value);
				offset = newOffset;
			}
			return { value: arr, newOffset: offset };
		}
		case GgufValueType.UINT64:
			return {
				value: Number(view.getBigUint64(offset, true)),
				newOffset: offset + 8,
			};
		case GgufValueType.INT64:
			return {
				value: Number(view.getBigInt64(offset, true)),
				newOffset: offset + 8,
			};
		case GgufValueType.FLOAT64:
			return { value: view.getFloat64(offset, true), newOffset: offset + 8 };
		default:
			throw new Error(`Unknown GGUF value type: ${type}`);
	}
}

function readTensorInfo(
	view: DataView,
	offset: number,
): { info: GgufTensorInfo; newOffset: number } {
	const { value: name, newOffset: afterName } = readString(view, offset);
	offset = afterName;
	const nDimensions = view.getUint32(offset, true);
	offset += 4;
	const dimensions: number[] = [];
	for (let i = 0; i < nDimensions; i++) {
		dimensions.push(Number(view.getBigUint64(offset, true)));
		offset += 8;
	}
	const type = view.getUint32(offset, true);
	offset += 4;
	const tensorOffset = Number(view.getBigUint64(offset, true));
	offset += 8;
	return {
		info: { name, nDimensions, dimensions, type, offset: tensorOffset },
		newOffset: offset,
	};
}

function getAlignment(metadata: Map<string, GgufKv>): number {
	const kv = metadata.get("general.alignment");
	if (kv && typeof kv.value === "number") return kv.value;
	return GGUF_DEFAULT_ALIGNMENT;
}

function alignTo(offset: number, alignment: number): number {
	return Math.ceil(offset / alignment) * alignment;
}

function calculateTotalDataSize(tensors: GgufTensorInfo[]): number {
	if (tensors.length === 0) return 0;
	let size = 0;
	for (const t of tensors) {
		const elemCount = t.dimensions.reduce((a, b) => a * b, 1);
		const typeSize = ggmlTypeSize(t.type);
		size = Math.max(size, t.offset + elemCount * typeSize);
	}
	return size;
}

function ggmlTypeSize(type: number): number {
	const sizes: Record<number, number> = {
		0: 4,
		1: 2,
		2: 0.5,
		3: 0.5,
		6: 1,
		7: 1,
	};
	return sizes[type] ?? 4;
}
