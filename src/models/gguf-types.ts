/** Magic number: "GGUF" encoded as uint32 little-endian = 0x46554747 */
export const GGUF_MAGIC = 0x46554747;

export const GGUF_VERSION = 3;

export const GGUF_DEFAULT_ALIGNMENT = 32;

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

export interface GgufHeader {
  magic: number;
  version: number;
  tensorCount: number;
  metadataKvCount: number;
}

export interface GgufKv {
  key: string;
  type: GgufValueType;
  isArray: boolean;
  value: unknown;
}

export interface GgufTensorInfo {
  name: string;
  nDimensions: number;
  dimensions: number[];
  type: number;
  offset: number;
}

export interface GgufContext {
  header: GgufHeader;
  metadata: Map<string, GgufKv>;
  tensors: GgufTensorInfo[];
  alignment: number;
  dataOffset: number;
  totalDataSize: number;
}
