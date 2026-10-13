/**
 * Two languages, one page. Turkish is the default for a Turkish browser;
 * everyone else gets English. The toggle in the header overrides both and
 * sticks in localStorage.
 *
 * Keys are flat on purpose: the page is small, and a flat map is one grep away
 * from showing every string it can say.
 */

export type Lang = 'tr' | 'en';

export const en = {
  'page.title': 'emergence — a language model trains in your browser',
  'page.description':
    'A dependency-free transformer, autodiff engine and training loop that train live in your browser. Watch a character-level language model emerge from noise: loss curve, attention heads, embedding space and next-character distribution, all recomputed every step.',

  'brand.tagline':
    'A transformer, an autodiff engine and a training loop — written from scratch, no dependencies — running live in this tab.',

  'stat.step': 'step',
  'stat.loss': 'loss',
  'stat.val': 'val',
  'stat.lr': 'lr',
  'stat.tps': 'tok/s',
  'stat.params': 'params',
  'stat.cost': 'step cost',

  'panel.loss': 'Loss',
  'panel.loss.em': 'train vs held-out',
  'panel.next': 'Next character',
  'panel.next.em': 'live distribution',
  'panel.attn': 'Attention',
  'panel.attn.em': 'layer',
  'panel.space': 'Embedding space',
  'panel.space.em': 'PCA of the learned vectors',
  'panel.sample': 'What it writes right now',
  'panel.tape': 'Forward graph',
  'panel.tape.em': 'cost per node, this configuration',
  'panel.grad': 'Gradient flow',
  'panel.grad.em': '‖∂L/∂θ‖ per tensor',
  'panel.controls': 'Controls',

  'prompt.aria': 'prompt',
  'btn.sample': 'sample',
  'sample.idle': 'start training to generate text…',
  'sample.warming': 'warming up the first sample…',

  'ctrl.arch': 'architecture',
  'ctrl.arch.gpt': 'transformer (gpt)',
  'ctrl.arch.mlp': 'fixed-window MLP',
  'ctrl.arch.bigram': 'bigram table',
  'ctrl.size': 'size',
  'ctrl.size.tiny': 'tiny — 28k params',
  'ctrl.size.small': 'small — 200k params',
  'ctrl.size.medium': 'medium — 700k params',
  'ctrl.corpus': 'corpus',
  'ctrl.corpus.shakespeare': 'tiny shakespeare (1.1 MB)',
  'ctrl.corpus.paste': 'your own text',
  'ctrl.lr': 'learning rate',
  'ctrl.warm': 'warm start',
  'ctrl.warm.none': 'from random noise',
  'ctrl.warm.ckpt': 'from the bundled checkpoint',
  'btn.train': 'train',
  'btn.pause': 'pause',
  'btn.reset': 'reset',
  'btn.download': 'download weights',
  'corpus.placeholder':
    'paste any text here — the tokenizer is rebuilt on it, and the model learns your text from scratch',
  'corpus.pasted': 'pasted text',
  'corpus.shakespeare': 'tiny shakespeare',

  'hint.loss':
    '{step}/{planned} steps · {stepMs}/step · {elapsed} of compute · learned on {chars} characters · {seen} characters processed so far',
  'hint.next':
    'after "{prompt}" — the entire vocabulary is the {vocab} distinct characters in the corpus',
  'hint.attn': 'row i = where position i looked while predicting character i+1 · {heads} heads total',
  'hint.attn.none': 'attention exists only in the transformer — switch architecture to see it',
  'hint.space':
    'learned embeddings of all {vocab} characters, projected to 2d (power iteration PCA) · {dim}-dimensional vectors · orange = whitespace',
  'hint.sample':
    'sampled from the live weights at step {step} · temperature 0.8, top-k 40 · the text above is regenerated as training continues',
  'hint.tape':
    'forward pass at context {block} · {flops} FLOPs per step (forward + backward) · {rate} FLOPs/s',
  'hint.grad':
    'per-tensor gradient RMS, log scale · global grad norm is clipped to 1.0 · highest: {top}',

  'footer.body':
    'Every number on this page comes from the code in src/: no ML library, no bundle of pretrained weights, no GPU. Same code, same corpus, same seed, from the command line:',
  'footer.repo': 'MIT © Emir Mızraklı',
} as const;

export type Key = keyof typeof en;

const tr: Record<Key, string> = {
  'page.title': 'emergence — tarayıcında bir dil modeli eğitiliyor',
  'page.description':
    'Sıfırdan yazılmış, hiç kütüphane kullanmayan bir transformer, otomatik türev motoru ve eğitim döngüsü — tam bu sekmede canlı çalışıyor. Harf seviyesinde bir dil modeli gözünün önünde doğuyor: kayıp eğrisi, attention başları, gömme uzayı ve sıradaki harf dağılımı her adımda yeniden hesaplanıyor.',

  'brand.tagline':
    'Sıfırdan yazılmış bir transformer, otomatik türev motoru ve eğitim döngüsü — hiç kütüphane yok — tam bu sekmede canlı çalışıyor.',

  'stat.step': 'adım',
  'stat.loss': 'kayıp',
  'stat.val': 'doğrulama',
  'stat.lr': 'öğr. hızı',
  'stat.tps': 'token/s',
  'stat.params': 'parametre',
  'stat.cost': 'adım maliyeti',

  'panel.loss': 'Kayıp',
  'panel.loss.em': 'eğitim vs görülmemiş veri',
  'panel.next': 'Sıradaki harf',
  'panel.next.em': 'canlı olasılık dağılımı',
  'panel.attn': 'Dikkat (attention)',
  'panel.attn.em': 'katman',
  'panel.space': 'Gömme uzayı',
  'panel.space.em': 'öğrenilen vektörlerin PCA projeksiyonu',
  'panel.sample': 'Şu anda ne yazıyor',
  'panel.tape': 'İleri geçiş grafiği',
  'panel.tape.em': 'bu konfigürasyonda düğüm başına maliyet',
  'panel.grad': 'Gradyan akışı',
  'panel.grad.em': 'tensör başına ‖∂L/∂θ‖',
  'panel.controls': 'Kontroller',

  'prompt.aria': 'başlangıç metni',
  'btn.sample': 'örnekle',
  'sample.idle': 'metin üretmesi için eğitimi başlat…',
  'sample.warming': 'ilk örnek hazırlanıyor…',

  'ctrl.arch': 'mimari',
  'ctrl.arch.gpt': 'transformer (gpt)',
  'ctrl.arch.mlp': 'sabit pencereli MLP',
  'ctrl.arch.bigram': 'bigram tablosu',
  'ctrl.size': 'boyut',
  'ctrl.size.tiny': 'minik — 28 bin parametre',
  'ctrl.size.small': 'küçük — 200 bin parametre',
  'ctrl.size.medium': 'büyük — 700 bin parametre',
  'ctrl.corpus': 'veri seti',
  'ctrl.corpus.shakespeare': 'tiny shakespeare (1.1 MB)',
  'ctrl.corpus.paste': 'kendi metnin',
  'ctrl.lr': 'öğrenme hızı',
  'ctrl.warm': 'hazır başlangıç',
  'ctrl.warm.none': 'sıfırdan (rastgele ağırlıklar)',
  'ctrl.warm.ckpt': 'paketlenmiş checkpoint ile',
  'btn.train': 'eğit',
  'btn.pause': 'duraklat',
  'btn.reset': 'sıfırla',
  'btn.download': 'ağırlıkları indir',
  'corpus.placeholder':
    'buraya istediğin metni yapıştır — tokenizer ona göre yeniden kurulur ve model senin metnini sıfırdan öğrenir',
  'corpus.pasted': 'yapıştırılan metin',
  'corpus.shakespeare': 'tiny shakespeare',

  'hint.loss':
    '{step}/{planned} adım · adım başına {stepMs} · {elapsed} işlem · {chars} karakter öğrenildi · şu ana kadar {seen} karakter işlendi',
  'hint.next':
    '"{prompt}" sonrası — tüm sözlük, veri setindeki {vocab} farklı karakterden oluşuyor',
  'hint.attn': 'i. satır = i. karakteri tahmin ederken i. pozisyonun nereye baktığı · toplam {heads} head',
  'hint.attn.none': 'attention yalnızca transformer içinde var — mimariyi değiştirip gör',
  'hint.space':
    '{vocab} karakterin öğrenilmiş vektörleri, 2 boyuta indirgenmiş (power iteration PCA) · {dim} boyutlu vektörler · turuncu = boşluk',
  'hint.sample':
    'canlı ağırlıklardan, {step}. adımda örneklendi · temperature 0.8, top-k 40 · eğitim sürdükçe yukarıdaki metin yeniden üretilir',
  'hint.tape':
    '{block} bağlamında ileri geçiş · adım başına {flops} FLOP (ileri + geri) · saniyede {rate} FLOP',
  'hint.grad':
    'tensör başına gradyan RMS, log ölçek · global gradyan normu 1.0 değerine kırpılıyor · en yüksek: {top}',

  'footer.body':
    'Bu sayfadaki her sayı src/ içindeki koddan geliyor: ML kütüphanesi yok, hazır ağırlık yok, GPU yok. Aynı kod, aynı veri seti, aynı tohum, komut satırından:',
  'footer.repo': 'MIT © Emir Mızraklı',
};

const DICT: Record<Lang, Record<Key, string>> = { en, tr };

let current: Lang = 'en';

/** Turkish browsers get Turkish, everyone else English, and a choice sticks. */
export function detectLang(): Lang {
  try {
    const stored = localStorage.getItem('emergence-lang');
    if (stored === 'tr' || stored === 'en') return stored;
  } catch {
    /* private mode: no storage, no problem */
  }
  return (navigator.language || 'en').toLowerCase().startsWith('tr') ? 'tr' : 'en';
}

export function lang(): Lang {
  return current;
}

export function t(key: Key, vars?: Record<string, string | number>): string {
  let out = DICT[current][key] ?? en[key] ?? key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) out = out.replaceAll(`{${k}}`, String(v));
  }
  return out;
}

/** Fill every static string in the document from the current dictionary. */
export function applyStatic(): void {
  document.documentElement.lang = current;
  document.title = t('page.title');
  document.querySelector('meta[name="description"]')?.setAttribute('content', t('page.description'));

  for (const el of document.querySelectorAll<HTMLElement>('[data-i18n]')) {
    el.textContent = t(el.dataset.i18n as Key);
  }
  for (const el of document.querySelectorAll<HTMLElement>('[data-i18n-aria]')) {
    el.setAttribute('aria-label', t(el.dataset.i18nAria as Key));
  }
  for (const el of document.querySelectorAll<HTMLElement>('[data-i18n-placeholder]')) {
    el.setAttribute('placeholder', t(el.dataset.i18nPlaceholder as Key));
  }

  const toggle = document.getElementById('lang');
  if (toggle) {
    toggle.textContent = current === 'tr' ? 'EN' : 'TR';
    toggle.title = current === 'tr' ? 'switch to English' : "Türkçe'ye geç";
  }
}

export function setLang(next: Lang): void {
  current = next;
  try {
    localStorage.setItem('emergence-lang', next);
  } catch {
    /* ignore */
  }
  applyStatic();
}

export function initLang(): void {
  current = detectLang();
  applyStatic();
}

export function toggleLang(): Lang {
  setLang(current === 'tr' ? 'en' : 'tr');
  return current;
}
