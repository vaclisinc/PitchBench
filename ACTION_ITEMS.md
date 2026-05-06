# ACTION ITEMS!

| Task Name | Description | Assignee | Priority | Finished? |
|-----------|-------------|----------|----------|-----------|
| Sample N per experiment | Randomly sample a fixed N stimuli per experiment so all exps use the same sample size — 1. fairer comparison for different abilities 2. without running all stimuli | Jimmy | | |
| Add sampler / more audio sources | Extend the audio engine to support multiple additional audio sources via a sampler | X | | |
| Structured output support | Replace prompt-level output restrictions with proper structured output so model responses are reliably parseable | Jimmy | | |
| Convolution support | Convolve stimuli with frequency responses of real places (room IRs); consider C++ + fast_conv for speed to replace the existing café implementation | Jimmy | | |
| LLM cost tracking (OpenRouter) | Track token usage and cost per run for OpenRouter models; surface in result files and summary | Jimmy | | |
parse open response for doremi by cheap llm model
fix the Strata bug, if sample < source * note, may return error
  ---

## Models to run after finalizing experiments

| Model | Notes | Access |
|-------|-------|--------|
| Flamingo Next | Best candidate? | local / weights TBD |
| Music Flamingo | Already wired, baseline reference | local |
| Qwen 3.5-Omni | Not on OpenRouter | [Alibaba Cloud API](https://www.alibabacloud.com/help/en/model-studio/get-api-key) |
| Qwen 3.6 | Unclear if music/audio support exists | TBD |
| Gemini 3.1 Pro | `google/gemini-3.1-pro-preview` | [OpenRouter](https://openrouter.ai/google/gemini-3.1-pro-preview) |
| GPT-4o Audio | `openai/gpt-4o-audio-preview` | [OpenRouter](https://openrouter.ai/openai/gpt-4o-audio-preview) |
| GPT Audio | `openai/gpt-audio` | [OpenRouter](https://openrouter.ai/openai/gpt-audio) · [OpenAI Docs](https://developers.openai.com/api/docs/models/gpt-audio) |
| ElevenLabs | | [API](https://elevenlabs.io/api) |
| Sesame CSM-1B | | [HuggingFace](https://huggingface.co/sesame/csm-1b) · [csm1b.com](https://csm1b.com/) |
| Kimi Audio 7B | | [HuggingFace](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct) |

---

