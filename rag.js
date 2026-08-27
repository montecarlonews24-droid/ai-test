/* rag.js — RAG محلي بدون أي مكتبة خارجية وبدون أي تنزيل إضافي
 *
 * التركيب في index.html (3 أسطر فقط):
 *
 *   import { RAG, ragPrompt, RAG_SYSTEM_PROMPT } from "./rag.js";
 *   const rag = new RAG(); await rag.init();
 *   // قبل الإرسال للنموذج:
 *   const { context } = await rag.buildContext(userQuestion);
 *   const finalUserMsg = context ? ragPrompt(context, userQuestion) : userQuestion;
 *
 * ولو فيه مستندات، استبدل السيستم برومبت بـ RAG_SYSTEM_PROMPT.
 *
 * البحث: BM25 — بحث كلمات موزون، JS خالص. مناسب لعشرات/مئات المقاطع.
 * التخزين: IndexedDB. المقاطع والفهرس يُبنيان في الذاكرة عند الإقلاع (سريع).
 */

/* ============================ 1. تطبيع النص ============================ */

const AR_DIACRITICS = /[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]/g;
const TATWEEL = /\u0640/g;

/** توحيد شكل الكلمة العربية حتى ينجح البحث. بدون هذا البحث العربي يفشل. */
export function normalize(s) {
  return String(s || "")
    .replace(AR_DIACRITICS, "")
    .replace(TATWEEL, "")
    .replace(/[\u0622\u0623\u0625\u0671]/g, "\u0627") // آ أ إ ٱ → ا
    .replace(/\u0629/g, "\u0647")                     // ة → ه
    .replace(/[\u0649]/g, "\u064A")                   // ى → ي
    .replace(/\u0624/g, "\u0648")                     // ؤ → و
    .replace(/\u0626/g, "\u064A")                     // ئ → ي
    .replace(/[\u0660-\u0669]/g, d => String(d.charCodeAt(0) - 0x0660)) // أرقام عربية
    .replace(/[\u06F0-\u06F9]/g, d => String(d.charCodeAt(0) - 0x06F0))
    .toLowerCase()
    .replace(/[\u200B-\u200F\u202A-\u202E]/g, "")
    .normalize("NFKC");
}

const STOP = new Set([
  // عربي
  "في","من","على","الى","إلى","عن","مع","هذا","هذه","ذلك","التي","الذي","ما","لا","ان","أن",
  "إن","كان","كانت","هو","هي","هم","قد","كل","بعد","قبل","او","أو","ثم","حتى","لكن","بين",
  "عند","هناك","يكون","تكون","به","له","بها","لها","و","يا","اي","أي",
  // إنجليزي
  "the","a","an","of","to","in","is","are","was","were","and","or","for","on","at","by","it",
  "this","that","with","as","be","from","but","not","have","has","had","you","i","we","they",
  "what","how","why","when","which","do","does","did","can","will","would","there","their"
]);

function tokenize(s) {
  return normalize(s)
    .split(/[^\p{L}\p{N}_]+/u)
    .filter(w => w.length > 1 && !STOP.has(w));
}

/** تقدير عدد التوكنات. العربي أغلى بكثير من الإنجليزي عند نفس عدد الكلمات. */
export function estTokens(s) {
  const t = String(s || "");
  const arWords = (t.match(/[\u0600-\u06FF]+/g) || []).length;
  const rest = t.replace(/[\u0600-\u06FF]+/g, " ").split(/\s+/).filter(Boolean).length;
  return Math.ceil(arWords * 2.6 + rest * 1.35) || 0;
}

/* ============================ 2. التقطيع ============================ */

const SENT_SPLIT = /(?<=[.!?؟。…]|[۔؛;]|\n)\s+/;

/**
 * يقطّع النص لمقاطع بحدود جُمَل، مع تداخل حتى لا تنقطع المعلومة بين مقطعين.
 */
export function chunkText(text, { size = 200, overlap = 40 } = {}) {
  const clean = String(text || "")
    .replace(/\r/g, "")
    .replace(/```[\s\S]*?```/g, m => m)      // خلي الكود كما هو
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  if (!clean) return [];

  const parts = clean.split(SENT_SPLIT).map(s => s.trim()).filter(Boolean);
  const chunks = [];
  let buf = [], bufTok = 0;

  const flush = () => {
    if (!buf.length) return;
    chunks.push(buf.join(" ").trim());
    // تداخل: احتفظ بآخر جُمَل بما يقارب overlap توكن
    let keep = [], k = 0;
    for (let i = buf.length - 1; i >= 0; i--) {
      const t = estTokens(buf[i]);
      if (k + t > overlap) break;
      keep.unshift(buf[i]); k += t;
    }
    buf = keep; bufTok = k;
  };

  for (const p of parts) {
    const t = estTokens(p);
    if (t > size * 1.6) {                    // جملة عملاقة: قسّمها بالكلمات
      flush(); if (buf.length) { chunks.push(buf.join(" ")); buf = []; bufTok = 0; }
      const words = p.split(/\s+/);
      let acc = [], at = 0;
      for (const w of words) {
        const wt = estTokens(w);
        if (at + wt > size && acc.length) { chunks.push(acc.join(" ")); acc = []; at = 0; }
        acc.push(w); at += wt;
      }
      if (acc.length) chunks.push(acc.join(" "));
      continue;
    }
    if (bufTok + t > size && buf.length) flush();
    buf.push(p); bufTok += t;
  }
  if (buf.length) chunks.push(buf.join(" ").trim());

  return chunks.filter(c => estTokens(c) >= 8);
}

/* ============================ 3. فهرس BM25 ============================ */

class BM25 {
  constructor(k1 = 1.2, b = 0.75) { this.k1 = k1; this.b = b; this.reset(); }

  reset() { this.docs = []; this.df = new Map(); this.avgdl = 0; }

  build(chunks) {
    this.reset();
    for (const c of chunks) {
      const terms = tokenize(c.text);
      const tf = new Map();
      for (const t of terms) tf.set(t, (tf.get(t) || 0) + 1);
      for (const t of tf.keys()) this.df.set(t, (this.df.get(t) || 0) + 1);
      this.docs.push({ ...c, tf, len: terms.length, norm: normalize(c.text) });
    }
    const total = this.docs.reduce((s, d) => s + d.len, 0);
    this.avgdl = this.docs.length ? total / this.docs.length : 0;
  }

  search(query, topK = 2) {
    if (!this.docs.length) return [];
    const qTerms = [...new Set(tokenize(query))];
    if (!qTerms.length) return [];
    const N = this.docs.length;
    const qNorm = normalize(query);

    const scored = this.docs.map(d => {
      let score = 0, hits = 0;
      for (const t of qTerms) {
        const f = d.tf.get(t);
        if (!f) continue;
        hits++;
        const n = this.df.get(t) || 0;
        const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
        score += idf * (f * (this.k1 + 1)) /
                 (f + this.k1 * (1 - this.b + this.b * (d.len / (this.avgdl || 1))));
      }
      // مكافأة صغيرة لتغطية أكثر كلمات السؤال، ولوجود العبارة حرفياً
      if (hits) score *= (0.6 + 0.4 * (hits / qTerms.length));
      if (qNorm.length > 8 && d.norm.includes(qNorm)) score *= 1.3;
      return { ...d, score };
    });

    return scored
      .filter(d => d.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}

/* ============================ 4. التخزين (IndexedDB) ============================ */

const DB_NAME = "rag-db", DB_VER = 1, STORE = "docs";

function openDB() {
  return new Promise((res, rej) => {
    const req = indexedDB.open(DB_NAME, DB_VER);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error);
  });
}

function tx(db, mode, fn) {
  return new Promise((res, rej) => {
    const t = db.transaction(STORE, mode);
    const store = t.objectStore(STORE);
    const out = fn(store);
    t.oncomplete = () => res(out?.result ?? out);
    t.onerror = () => rej(t.error);
  });
}

/* ============================ 5. الواجهة الرئيسية ============================ */

export class RAG {
  /**
   * @param {object} opts
   *  chunkSize      حجم المقطع بالتوكن (200 افتراضياً)
   *  overlap        تداخل بالتوكن (40)
   *  topK           كم مقطع يدخل السياق (2 — لا تزيدها قبل ما تقيس prefill)
   *  maxContextTok  سقف السياق المسترجع (450)
   *  minScore       أقل درجة تُقبل، تحتها نعتبر "ما لقينا" (0.6)
   */
  constructor(opts = {}) {
    this.o = {
      chunkSize: 200, overlap: 40, topK: 2,
      maxContextTok: 450, minScore: 0.6, ...opts
    };
    this.db = null;
    this.docs = [];        // [{id,name,text,addedAt}]
    this.index = new BM25();
    this.ready = false;
  }

  async init() {
    this.db = await openDB();
    this.docs = (await tx(this.db, "readonly", s => s.getAll())) || [];
    this._reindex();
    this.ready = true;
    return this;
  }

  _reindex() {
    const chunks = [];
    for (const d of this.docs) {
      chunkText(d.text, { size: this.o.chunkSize, overlap: this.o.overlap })
        .forEach((text, i) => chunks.push({ id: `${d.id}#${i}`, docId: d.id, source: d.name, text }));
    }
    this.index.build(chunks);
    this.chunkCount = chunks.length;
  }

  /** إضافة نص. name = اسم المصدر الذي سيظهر للمستخدم. */
  async addText(name, text) {
    const doc = {
      id: `d_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      name: String(name || "بدون اسم"),
      text: String(text || ""),
      addedAt: Date.now()
    };
    if (!doc.text.trim()) throw new Error("النص فارغ");
    await tx(this.db, "readwrite", s => s.put(doc));
    this.docs.push(doc);
    this._reindex();
    return doc;
  }

  /** إضافة ملف من <input type="file"> — يدعم .txt و .md و .json */
  async addFile(file) {
    const text = await file.text();
    if (file.name.endsWith(".json")) {
      const data = JSON.parse(text);
      const arr = Array.isArray(data) ? data : [data];
      const added = [];
      for (const it of arr) {
        const body = it.text ?? it.content ?? it.body;
        if (body) added.push(await this.addText(it.title || it.name || file.name, body));
      }
      return added;
    }
    return [await this.addText(file.name, text)];
  }

  /** تحميل docs.json من نفس النطاق (للمعرفة الثابتة داخل المستودع). */
  async addFromUrl(url = "./docs.json") {
    const r = await fetch(url, { cache: "no-cache" });
    if (!r.ok) throw new Error(`تعذّر تحميل ${url} (${r.status})`);
    const data = await r.json();
    const arr = Array.isArray(data) ? data : [data];
    const have = new Set(this.docs.map(d => d.name));
    const added = [];
    for (const it of arr) {
      const name = it.title || it.name || url;
      const body = it.text ?? it.content ?? it.body;
      if (body && !have.has(name)) added.push(await this.addText(name, body));
    }
    return added;
  }

  list() {
    return this.docs.map(d => ({
      id: d.id, name: d.name, chars: d.text.length,
      tokens: estTokens(d.text), addedAt: d.addedAt
    }));
  }

  async remove(id) {
    await tx(this.db, "readwrite", s => s.delete(id));
    this.docs = this.docs.filter(d => d.id !== id);
    this._reindex();
  }

  async clear() {
    await tx(this.db, "readwrite", s => s.clear());
    this.docs = [];
    this._reindex();
  }

  /**
   * القلب: يبني السياق من المقاطع الأعلى صلة، ضمن سقف التوكنات.
   * يرجّع { context:"", sources:[], hits:[] } إذا ما في نتيجة كافية.
   */
  async buildContext(question, opts = {}) {
    const { topK = this.o.topK, maxContextTok = this.o.maxContextTok,
            minScore = this.o.minScore } = opts;

    if (!this.ready) await this.init();
    const hits = this.index.search(question, topK);
    if (!hits.length || hits[0].score < minScore) {
      return { context: "", sources: [], hits: [], tokens: 0 };
    }

    const picked = [];
    let used = 0;
    for (const h of hits) {
      const t = estTokens(h.text);
      if (used + t > maxContextTok && picked.length) break;
      picked.push(h); used += t;
    }

    const context = picked
      .map((h, i) => `[${i + 1}] ${h.source}\n${h.text}`)
      .join("\n\n");

    return {
      context,
      sources: [...new Set(picked.map(h => h.source))],
      hits: picked.map(h => ({ source: h.source, score: +h.score.toFixed(3) })),
      tokens: used
    };
  }
}

/* ============================ 6. البرومبت ============================ */

/** استبدل السيستم برومبت بهذا عندما يكون هناك مستندات. */
export const RAG_SYSTEM_PROMPT = `You are a small offline assistant running on the user's phone.

You will be given CONTEXT extracted from the user's own documents.

Rules:
- Answer ONLY from the CONTEXT. Do not use your own knowledge.
- If the CONTEXT does not contain the answer, say exactly: "Not in your documents."
- Be brief: 1-3 sentences unless asked for detail. Never pad.
- Do not invent names, numbers, or dates that are not in the CONTEXT.
- Reply in English.`;

/** يغلّف السؤال مع السياق. استخدمه كرسالة user. */
export function ragPrompt(context, question) {
  return `CONTEXT:\n${context}\n\nQUESTION: ${question}\n\nAnswer from the CONTEXT only.`;
}

/* ============================ 7. أداة قياس (اختيارية) ============================ */

/** قِس كلفة الـ prefill عندك: مرّرها دالة تُرسل للنموذج وترجّع Promise. */
export async function measurePrefill(sendFn, context) {
  const t0 = performance.now();
  await sendFn(ragPrompt(context, "Reply with the single word: ok"));
  return Math.round(performance.now() - t0);
}

export default RAG;
