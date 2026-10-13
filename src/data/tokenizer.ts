/**
 * Character-level tokenizer.
 *
 * Character level keeps the whole pipeline inspectable: the vocabulary is the
 * set of characters actually present in the corpus (usually 40–90 symbols), so
 * "token id 37" is a character you can point at, and the model has to learn
 * spelling before it can learn words. Each corpus gets its own tokenizer, which
 * is why pasting your own text means training a fresh model.
 */
export class Tokenizer {
  readonly chars: string[];
  readonly vocabSize: number;
  private toId: Map<string, number>;

  constructor(chars: string[]) {
    this.chars = chars;
    this.vocabSize = chars.length;
    this.toId = new Map(chars.map((c, i) => [c, i]));
  }

  static fromText(text: string, extra = ''): Tokenizer {
    const set = new Set<string>();
    for (const ch of text) set.add(ch);
    for (const ch of extra) set.add(ch);
    const chars = [...set].sort((a, b) => a.codePointAt(0)! - b.codePointAt(0)!);
    return new Tokenizer(chars);
  }

  /** Encode text; characters outside the vocabulary become `fallback`. */
  encode(text: string, fallback = 0): Int32Array {
    const out = new Int32Array(text.length);
    for (let i = 0; i < text.length; i++) {
      const id = this.toId.get(text[i]);
      out[i] = id === undefined ? fallback : id;
    }
    return out;
  }

  decode(ids: ArrayLike<number>): string {
    let s = '';
    for (let i = 0; i < ids.length; i++) {
      const c = this.chars[ids[i]];
      if (c !== undefined) s += c;
    }
    return s;
  }

  /** Fraction of a string's characters that are inside the vocabulary. */
  coverage(text: string): number {
    if (text.length === 0) return 1;
    let known = 0;
    for (const ch of text) if (this.toId.has(ch)) known++;
    return known / text.length;
  }

  toJSON(): { chars: string[] } {
    return { chars: this.chars };
  }

  static fromJSON(json: { chars: string[] }): Tokenizer {
    return new Tokenizer(json.chars);
  }
}
