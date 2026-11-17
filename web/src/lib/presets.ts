import type { Settings } from './api'

export const DEFAULTS: Settings = {
  engine: 'xtts', lang: 'en', speed: 1, gap: 0.25, paraGap: 0.6, fit: false, duration: 30,
  temperature: 0.75, topP: 0.85, topK: 50, repetitionPenalty: 10, seed: '', refSeconds: 30,
  pitch: 0, gainDb: 0, normalize: false, trim: true, casual: false, indianText: false, techTerms: false, smartPace: false, pronunciations: '',
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
  'Course lesson': { speed: 0.95, gap: 0.35, paraGap: 0.9, temperature: 0.6, normalize: true, smartPace: true, techTerms: true, indianText: true },
  'Friendly Indian explainer': { speed: 1.0, gap: 0.25, paraGap: 0.7, temperature: 0.8, casual: true, normalize: true, smartPace: true, techTerms: true, indianText: true },
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

// Fill-in-the-blank scripts. Replace the <angle brackets> with your own content.
export const TEMPLATES: { label: string; text: string }[] = [
  {
    label: 'Course lesson',
    text: `Welcome back, everyone. [pause 0.5s]

In this lesson, we are going to learn about <topic>. By the end, you will be able to <outcome>.

First, let's understand what <topic> is. [speed 0.9] <Topic> is <simple definition in one sentence>. [speed 1.0] [pause 0.4s]

Let's take a simple example. <Example in two or three short sentences.>

Now, why does this matter? <Real-life reason, for example a job or project use.>

Let's quickly recap. [pause 0.3s] One, <point one>. Two, <point two>. Three, <point three>.

In the next lesson, we will look at <next topic>. Thank you, and see you there.`,
  },
  {
    label: 'Friendly Indian explainer',
    text: `Okay, so let's talk about <topic>. [pause 0.4s] Don't worry, it is not as difficult as it sounds.

Basically, <topic> is just <simple idea>. Think of it like <everyday analogy>, you know?

See, here's the thing. <Key point in one short sentence.> Makes sense, right? [pause 0.4s]

Let's see an example. <Example.> Simple, no?

So remember, <one-line takeaway>. Okay, let's move on.`,
  },
  {
    label: 'Hinglish explainer',
    text: `[lang hi] हेलो दोस्तो, [pause 0.3s] आज हम बात करेंगे [lang en] <topic> [lang hi] के बारे में। [lang en]

[lang hi] देखो, ये बहुत आसान है। [lang en] <Topic> is basically <simple idea in English>.

[lang hi] समझ आया? [lang en] [pause 0.4s] Let's see an example. <Example.>

[lang hi] चलो, अब अगले हिस्से पर चलते हैं। [lang en]`,
  },
  {
    label: 'Quick recap',
    text: `Quick recap. [pause 0.4s] Today we covered <topic>. [pause 0.3s]

One, <point one>. Two, <point two>. Three, <point three>.

If you remember just one thing, remember this: [pause 0.4s] <key takeaway>.`,
  },
]

// Indian-style conversational snippets. Hindi phrases are in Devanagari with [lang] markers,
// because Hindi in Roman letters is read with English sounds.
export const PHRASES: { label: string; text: string }[] = [
  { label: 'Okay, so…', text: "Okay, so let's start." },
  { label: 'Basically…', text: 'Basically, the idea is very simple.' },
  { label: 'See, here\'s the thing', text: "See, here's the thing." },
  { label: 'Don\'t worry', text: "Don't worry, it is easy." },
  { label: 'Makes sense, right?', text: 'Makes sense, right?' },
  { label: 'Think of it like…', text: 'Think of it like this.' },
  { label: 'चलो शुरू करते हैं', text: '[lang hi] चलो, शुरू करते हैं। [lang en]' },
  { label: 'समझ आया?', text: '[lang hi] समझ आया? [lang en]' },
  { label: 'देखो…', text: '[lang hi] देखो, बहुत आसान है। [lang en]' },
  { label: 'कोई बात नहीं', text: '[lang hi] कोई बात नहीं, आराम से सीखो। [lang en]' },
  { label: 'अब आगे बढ़ते हैं', text: '[lang hi] अब आगे बढ़ते हैं। [lang en]' },
]

export const SLANG_EXAMPLES = `IdentityIQ = Identity I Q
SQL = sequel
Kubernetes = koo-ber-net-ees
bro = bruh`
