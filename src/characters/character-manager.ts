import type { CharacterConfig } from "./character.js";
import { Character } from "./character.js";

/**
 * CRUD lifecycle manager for Character instances.
 *
 * Centralizes creation, retrieval, removal, and bulk shutdown of characters.
 */
export class CharacterManager {
	private characters: Map<string, Character> = new Map();

	/**
	 * Create and register a new character.
	 *
	 * @param config - Character configuration.
	 * @returns The newly created Character instance.
	 */
	create(config: CharacterConfig): Character {
		const char = new Character(config);
		this.characters.set(char.id, char);
		return char;
	}

	/**
	 * Retrieve a character by its ID.
	 *
	 * @param id - Character identifier.
	 * @returns The Character, or undefined if not found.
	 */
	get(id: string): Character | undefined {
		return this.characters.get(id);
	}

	/**
	 * @returns All registered characters.
	 */
	getAll(): Character[] {
		return Array.from(this.characters.values());
	}

	/**
	 * Stop and remove a character by ID.
	 *
	 * @param id - Character identifier to remove.
	 */
	async remove(id: string): Promise<void> {
		const char = this.characters.get(id);
		if (char) {
			char.stop();
			this.characters.delete(id);
		}
	}

	/** Stop all active characters without removing them. */
	stopAll(): void {
		for (const char of this.characters.values()) {
			char.stop();
		}
	}

	/** Stop all characters and remove them from the manager. */
	async clear(): Promise<void> {
		this.stopAll();
		this.characters.clear();
	}

	get count(): number {
		return this.characters.size;
	}

	get activeCount(): number {
		let count = 0;
		for (const char of this.characters.values()) {
			if (char.isActive) count++;
		}
		return count;
	}
}
