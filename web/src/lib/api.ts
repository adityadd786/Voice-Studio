export type Voice = { name: string; duration: number }
export type Analysis = {
  duration_s: number
  digital_silence_pct: number
  speech_pct: number
  snr_db: number
  loudness_db: number
  peak_db: number
  clipping_pct: number
  long_pauses: number
  pitch: { median_hz?: number; low_hz?: number; high_hz?: number; register?: string; variation?: number }
  tips: string[]
}
export type Dataset = { name: string; raw_minutes: number; segments: number; minutes: number; built: boolean }
export type ModelInfo = { name: string; size_mb: number }
export type Take = { file: string; seconds: number; seed: number }
export type Job = {
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  done: number
  total: number
  stage?: string
  // generation jobs return takes; dataset-build jobs return segment stats
  result: ({ takes: Take[]; elapsed: number } & { segments?: number; minutes?: number }) | null
  error: string | null
}

export type Settings = {
  engine: 'xtts' | 'f5'
  model: string // "base" or a fine-tuned model name
  lang: string
  speed: number
  gap: number
  paraGap: number
  fit: boolean
  duration: number
  temperature: number
  topP: number
  topK: number
  repetitionPenalty: number
  seed: string // empty = random
  refSeconds: number
  pitch: number
  gainDb: number
  normalize: boolean
  trim: boolean
  casual: boolean
  indianText: boolean
  techTerms: boolean
  smartPace: boolean
  pronunciations: string // "word = say" per line
  takes: number
  format: 'wav' | 'mp3'
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const d = body.detail
    throw new Error(typeof d === 'string' ? d : d ? JSON.stringify(d) : `Request failed (${res.status})`)
  }
  return res.json()
}

export function parsePronunciations(raw: string): [string, string][] {
  return raw
    .split('\n')
    .map((l) => l.split('=').map((s) => s.trim()))
    .filter((p): p is [string, string] => p.length >= 2 && !!p[0] && !!p[1])
    .map(([a, ...b]) => [a, b.join('=')])
}

export const api = {
  status: () => fetch('/api/status').then(json<{ cuda: boolean; gpu: string | null; langs: string[]; f5: boolean }>),
  voices: () => fetch('/api/voices').then(json<Voice[]>),
  addVoice: (name: string, file: File) => {
    const fd = new FormData()
    fd.append('name', name)
    fd.append('file', file)
    return fetch('/api/voices', { method: 'POST', body: fd }).then(json<Voice>)
  },
  analyze: (name: string) => fetch(`/api/voices/${encodeURIComponent(name)}/analysis`).then(json<Analysis>),
  cleanVoice: (name: string) =>
    fetch(`/api/voices/${encodeURIComponent(name)}/clean`, { method: 'POST' }).then(json<Voice>),
  renameVoice: (name: string, newName: string) =>
    fetch(`/api/voices/${encodeURIComponent(name)}/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName }),
    }).then(json<Voice>),
  deleteVoice: (name: string) =>
    fetch(`/api/voices/${encodeURIComponent(name)}`, { method: 'DELETE' }).then(json<{ ok: boolean }>),
  generate: (voice: string, text: string, s: Settings) =>
    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        voice, text, engine: s.engine, model: s.model, lang: s.lang, speed: s.speed, gap: s.gap, para_gap: s.paraGap,
        duration: s.fit ? s.duration : null, temperature: s.temperature, top_p: s.topP, top_k: s.topK,
        repetition_penalty: s.repetitionPenalty, seed: s.seed.trim() === '' ? null : Number(s.seed),
        ref_seconds: s.refSeconds, pitch: s.pitch, gain_db: s.gainDb, normalize: s.normalize,
        trim: s.trim, casual: s.casual, indian_text: s.indianText, tech_terms: s.techTerms, smart_pace: s.smartPace, pronunciations: parsePronunciations(s.pronunciations),
        takes: s.takes, format: s.format,
      }),
    }).then(json<{ job: string }>),
  // ---- fine-tuning datasets and models
  datasets: () => fetch('/api/datasets').then(json<Dataset[]>),
  createDataset: (name: string) =>
    fetch('/api/datasets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }).then(json<Dataset>),
  addDatasetAudio: (name: string, file: Blob, filename: string) => {
    const fd = new FormData()
    fd.append('file', file, filename)
    return fetch(`/api/datasets/${encodeURIComponent(name)}/audio`, { method: 'POST', body: fd }).then(json<Dataset>)
  },
  buildDataset: (name: string) =>
    fetch(`/api/datasets/${encodeURIComponent(name)}/build`, { method: 'POST' }).then(json<{ job: string }>),
  deleteDataset: (name: string) =>
    fetch(`/api/datasets/${encodeURIComponent(name)}`, { method: 'DELETE' }).then(json<{ ok: boolean }>),
  datasetZipUrl: (name: string) => `/api/datasets/${encodeURIComponent(name)}/download`,
  notebookUrl: () => '/api/training/notebook',
  models: () => fetch('/api/models').then(json<ModelInfo[]>),
  importModel: (name: string, file: File) => {
    const fd = new FormData()
    fd.append('name', name)
    fd.append('file', file)
    return fetch('/api/models', { method: 'POST', body: fd }).then(json<{ name: string }>)
  },
  deleteModel: (name: string) =>
    fetch(`/api/models/${encodeURIComponent(name)}`, { method: 'DELETE' }).then(json<{ ok: boolean }>),
  job: (id: string) => fetch(`/api/jobs/${id}`).then(json<Job>),
  cancel: (id: string) => fetch(`/api/jobs/${id}/cancel`, { method: 'POST' }).then(json<{ ok: boolean }>),
  audioUrl: (file: string) => `/api/audio/${encodeURIComponent(file)}`,
  voiceUrl: (name: string) => `/api/voices/${encodeURIComponent(name)}/audio`,
}
