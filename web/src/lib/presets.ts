import type { Settings } from './api'

export const DEFAULTS: Settings = {
  lang: 'en', speed: 1, gap: 0.25, paraGap: 0.6, fit: false, duration: 30,
  temperature: 0.75, topP: 0.85, topK: 50, repetitionPenalty: 10, seed: '', refSeconds: 30,
  pitch: 0, gainDb: 0, normalize: false, trim: true, casual: false, pronunciations: '',
  takes: 1, format: 'wav',
}

export const LANGS: Record<string, string> = {
  en: 'English', hi: 'Hindi', es: 'Spanish', fr: 'French', de: 'German', it: 'Italian',
  pt: 'Portuguese', pl: 'Polish', tr: 'Turkish', ru: 'Russian', nl: 'Dutch', cs: 'Czech',
  ar: 'Arabic', 'zh-cn': 'Chinese', ja: 'Japanese', ko: 'Korean', hu: 'Hungarian',
}

// Built-in presets. They only change the fields listed.
export const PRESETS: Record<string, Partial<Settings>> = {
  'Default': {},
  'Calm narration': { speed: 0.92, gap: 0.4, paraGap: 0.9, temperature: 0.6, normalize: true },
  'Audiobook': { speed: 0.95, gap: 0.45, paraGap: 1.1, temperature: 0.65, normalize: true, trim: true },
  'Energetic': { speed: 1.12, gap: 0.15, paraGap: 0.4, temperature: 0.9, topP: 0.9 },
  'Casual / slang': { speed: 1.05, gap: 0.2, temperature: 0.85, casual: true },
  'Tutorial / video': { speed: 1.0, gap: 0.3, paraGap: 0.7, temperature: 0.65, normalize: true },
  'Stable (fewer glitches)': { temperature: 0.5, topP: 0.75, topK: 30, repetitionPenalty: 12 },
}

export const MARKERS = [
  { label: 'Pause', text: '[pause 0.5s]' },
  { label: 'Slow', text: '[speed 0.85]' },
  { label: 'Fast', text: '[speed 1.15]' },
  { label: 'Normal', text: '[speed 1.0]' },
  { label: 'Pitch up', text: '[pitch +2]' },
  { label: 'Pitch down', text: '[pitch -2]' },
  { label: 'Language', text: '[lang hi]' },
  { label: 'Voice', text: '[voice name]' },
]

export const SLANG_EXAMPLES = `IdentityIQ = Identity I Q
SQL = sequel
Kubernetes = koo-ber-net-ees
bro = bruh`
