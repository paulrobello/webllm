/** Magic number identifying a valid GGUF file: "GGUF" as uint32 little-endian. */
export const GGUF_MAGIC = 0x46554747;

/** Supported GGUF specification version. */
export const GGUF_VERSION = 3;

/** Default byte alignment for tensor data sections. */
export const GGUF_DEFAULT_ALIGNMENT = 32;

/** Value types for GGUF metadata KV pairs. */
export enum GgufValueType {
	UINT8 = 0,
	INT8 = 1,
	UINT16 = 2,
	INT16 = 3,
	UINT32 = 4,
	INT32 = 5,
	FLOAT32 = 6,
	BOOL = 7,
	STRING = 8,
	ARRAY = 9,
	UINT64 = 10,
	INT64 = 11,
	FLOAT64 = 12,
}

/** Parsed GGUF file header containing format identification and section counts. */
export interface GgufHeader {
	/** File magic number, must match GGUF_MAGIC. */
	magic: number;
	/** GGUF specification version. */
	version: number;
	/** Number of tensor descriptors in the tensor info section. */
	tensorCount: number;
	/** Number of key-value pairs in the metadata section. */
	metadataKvCount: number;
}

/** A single metadata key-value entry from the GGUF metadata section. */
export interface GgufKv {
	/** Metadata key string. */
	key: string;
	/** Value type discriminator. */
	type: GgufValueType;
	/** Whether the value is an array type. */
	isArray: boolean;
	/** Deserialized value (type depends on GgufValueType). */
	value: unknown;
}

/** Descriptor for a single tensor stored in the GGUF data section. */
export interface GgufTensorInfo {
	/** Tensor name (e.g., "token_embd.weight"). */
	name: string;
	/** Number of dimensions. */
	nDimensions: number;
	/** Size of each dimension. */
	dimensions: number[];
	/** GGML tensor data type code. */
	type: number;
	/** Byte offset into the data section. */
	offset: number;
}

/** Complete parsed result of a GGUF binary file. */
export interface GgufContext {
	/** Parsed file header. */
	header: GgufHeader;
	/** Metadata key-value pairs keyed by metadata key string. */
	metadata: Map<string, GgufKv>;
	/** Descriptors for all tensors in the file. */
	tensors: GgufTensorInfo[];
	/** Byte alignment used for tensor data offsets. */
	alignment: number;
	/** Byte offset where tensor data begins in the buffer. */
	dataOffset: number;
	/** Total size in bytes of the tensor data section. */
	totalDataSize: number;
}
