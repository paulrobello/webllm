import type { CharacterConfig } from "./character.js";
import { Character } from "./character.js";

export class CharacterManager {
	private characters: Map<string, Character> = new Map();

	create(config: CharacterConfig): Character {
		const char = new Character(config);
		this.characters.set(char.id, char);
		return char;
	}

	get(id: string): Character | undefined {
		return this.characters.get(id);
	}

	getAll(): Character[] {
		return Array.from(this.characters.values());
	}

	async remove(id: string): Promise<void> {
		const char = this.characters.get(id);
		if (char) {
			char.stop();
			this.characters.delete(id);
		}
	}

	stopAll(): void {
		for (const char of this.characters.values()) {
			char.stop();
		}
	}

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
