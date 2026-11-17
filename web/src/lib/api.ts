export type Voice = { name: string; duration: number }
export type Take = { file: string; seconds: number; seed: number }
export type Job = {
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  done: number
  total: number
  result: { takes: Take[]; elapsed: number } | null
  error: string | null
}

export type Settings = {
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
  status: () => fetch('/api/status').then(json<{ cuda: boolean; gpu: string | null; langs: string[] }>),
  voices: () => fetch('/api/voices').then(json<Voice[]>),
  addVoice: (name: string, file: File) => {
    const fd = new FormData()
    fd.append('name', name)
    fd.append('file', file)
    return fetch('/api/voices', { method: 'POST', body: fd }).then(json<Voice>)
  },
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
        voice, text, lang: s.lang, speed: s.speed, gap: s.gap, para_gap: s.paraGap,
        duration: s.fit ? s.duration : null, temperature: s.temperature, top_p: s.topP, top_k: s.topK,
        repetition_penalty: s.repetitionPenalty, seed: s.seed.trim() === '' ? null : Number(s.seed),
        ref_seconds: s.refSeconds, pitch: s.pitch, gain_db: s.gainDb, normalize: s.normalize,
        trim: s.trim, casual: s.casual, indian_text: s.indianText, tech_terms: s.techTerms, smart_pace: s.smartPace, pronunciations: parsePronunciations(s.pronunciations),
        takes: s.takes, format: s.format,
      }),
    }).then(json<{ job: string }>),
  job: (id: string) => fetch(`/api/jobs/${id}`).then(json<Job>),
  cancel: (id: string) => fetch(`/api/jobs/${id}/cancel`, { method: 'POST' }).then(json<{ ok: boolean }>),
  audioUrl: (file: string) => `/api/audio/${encodeURIComponent(file)}`,
  voiceUrl: (name: string) => `/api/voices/${encodeURIComponent(name)}/audio`,
}
