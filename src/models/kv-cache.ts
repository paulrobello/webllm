/** Ring-buffer KV cache for per-layer GPU key/value tensor management. */

export interface KVCacheConfig {
	/** Number of transformer layers. */
	nLayers: number;
	/** Dimension of each key head. */
	nEmbdHeadK: number;
	/** Dimension of each value head. */
	nEmbdHeadV: number;
	/** Number of KV heads (supports GQA). */
	nKvHead: number;
	/** Maximum number of tokens in the cache (ring buffer size). */
	maxContextLength: number;
	/** Storage data type for K/V tensors. */
	dataType: "f32" | "f16";
}

/** Metadata for a single cell in the KV cache ring buffer. */
export interface KVCell {
	/** Token position in the sequence, or -1 if empty. */
	position: number;
	/** Owning sequence ID, or -1 if unoccupied. */
	sequenceId: number;
}

/**
 * Ring-buffer KV cache manager.
 *
 * Manages per-layer GPU buffers for key and value tensors during LLM inference.
 * Uses flat arrays with cell metadata tracking position and sequence occupancy,
 * following the same design as llama.cpp's KV cache.
 */
export class KVCache {
	private cells: KVCell[];
	private bufferSize: number;
	private head: number;
	private usedCells: number;
	private kBufferIds: (number | null)[];
	private vBufferIds: (number | null)[];

	constructor(config: KVCacheConfig) {
		this.bufferSize = config.maxContextLength;
		this.cells = Array.from({ length: this.bufferSize }, () => ({
			position: -1,
			sequenceId: -1,
		}));
		this.head = 0;
		this.usedCells = 0;
		this.kBufferIds = new Array(config.nLayers).fill(null);
		this.vBufferIds = new Array(config.nLayers).fill(null);
	}

	/** Find contiguous empty slots for `nTokens` tokens. */
	findSlots(nTokens: number, _sequenceId: number): number[] {
		if (nTokens > this.bufferSize) return [];

		const slots: number[] = [];
		const start = this.head;

		// Scan for contiguous empty cells starting from head
		for (let attempt = 0; attempt < this.bufferSize; attempt++) {
			const idx = (start + attempt) % this.bufferSize;
			const cell = this.cells[idx];
			if (cell.position === -1) {
				slots.push(idx);
				if (slots.length === nTokens) return slots;
			} else {
				slots.length = 0;
			}
		}

		// If no contiguous block from head, wrap from beginning
		slots.length = 0;
		for (let i = 0; i < this.bufferSize; i++) {
			if (this.cells[i].position === -1) {
				slots.push(i);
				if (slots.length === nTokens) return slots;
			} else {
				slots.length = 0;
			}
		}

		return [];
	}

	/** Write position and sequence metadata into allocated slots. */
	updateSlots(
		slotIndices: number[],
		positions: number[],
		sequenceId: number,
	): void {
		for (let i = 0; i < slotIndices.length; i++) {
			this.cells[slotIndices[i]].position = positions[i];
			this.cells[slotIndices[i]].sequenceId = sequenceId;
		}
		this.usedCells += slotIndices.length;
		const lastSlot = slotIndices[slotIndices.length - 1];
		this.head = (lastSlot + 1) % this.bufferSize;
	}

	/** Evict all cells belonging to a sequence, freeing its KV cache. */
	evictSequence(sequenceId: number): void {
		let freed = 0;
		for (const cell of this.cells) {
			if (cell.sequenceId === sequenceId) {
				cell.position = -1;
				cell.sequenceId = -1;
				freed++;
			}
		}
		this.usedCells -= freed;
	}

	/** Get the GPU buffer ID for the key tensor at a given layer. */
	getKBuffer(layer: number): number | null {
		return this.kBufferIds[layer] ?? null;
	}

	/** Get the GPU buffer ID for the value tensor at a given layer. */
	getVBuffer(layer: number): number | null {
		return this.vBufferIds[layer] ?? null;
	}

	/** Set the GPU buffer ID for the key tensor at a given layer. */
	setKBuffer(layer: number, bufferId: number): void {
		this.kBufferIds[layer] = bufferId;
	}

	/** Set the GPU buffer ID for the value tensor at a given layer. */
	setVBuffer(layer: number, bufferId: number): void {
		this.vBufferIds[layer] = bufferId;
	}

	/** Reset the cache to its initial empty state. */
	reset(): void {
		for (const cell of this.cells) {
			cell.position = -1;
			cell.sequenceId = -1;
		}
		this.head = 0;
		this.usedCells = 0;
		this.kBufferIds.fill(null);
		this.vBufferIds.fill(null);
	}

	/** Number of currently occupied cells. */
	get usedCellsCount(): number {
		return this.usedCells;
	}

	/** Total number of cells in the ring buffer. */
	get totalCells(): number {
		return this.bufferSize;
	}

	/** Fraction of buffer currently in use (0..1). */
	get utilizationRatio(): number {
		return this.usedCells / this.bufferSize;
	}
}
