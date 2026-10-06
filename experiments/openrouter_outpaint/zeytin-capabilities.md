# Zeytin — live Image API capabilities

Captured 6 October 2026. Native endpoint capabilities, not inferred from marketing names. All requests use 9:16. Quality selection belongs to the user.

| Model | Type | Native resolution | Native quality | References in this benchmark |
|---|---|---|---|---:|
| openai/gpt-image-2 | General generation and editing |  | auto, low, medium, high | 2 |
| google/gemini-3.1-flash-lite-image | Fast/cost-focused Gemini image tier | 1K |  | 2 |
| krea/krea-2-medium-turbo | Distilled speed-focused generation variant | 1K |  | 1 |
| krea/krea-2-medium | Balanced generation variant | 1K |  | 1 |
| krea/krea-2-large | Higher-capacity generation variant | 1K |  | 1 |
| microsoft/mai-image-2.5-pro | Quality-focused generation via Azure |  |  | 1 |
| qwen/qwen-image-3 | Unified generation and editing | 1K, 2K |  | 2 |
| qwen/qwen-image-3-pro | Pro generation and editing tier | 1K, 2K |  | 2 |
| x-ai/grok-imagine-image-2.0 | Generation/editing with quality and resolution controls | 1K, 2K | low, medium | 2 |
| bytedance-seed/seedream-5-0-pro | Precision/commercial editing tier | 1K, 2K |  | 2 |
| bytedance-seed/seedream-5-0-lite | Visual-reference generation tier | 2K, 4K |  | 2 |
| microsoft/mai-image-2.6 | Precision generation and editing tier |  |  | 2 |
| microsoft/mai-image-2.6-flash | Latency/cost-focused generation and editing tier |  |  | 2 |
| google/gemini-nano-banana-2.1 | Flash generation and editing tier | 1K, 2K, 4K |  | 2 |
| tencent/hy-image-v3.5-preview | Unified T2I, I2I and multi-turn editing | 1K, 1.5K, 2K, 4K |  | 2 |
| bytedance-seed/seedream-5-0-flash | Speed/cost-focused generation and editing tier | 1K, 2K |  | 2 |
| black-forest-labs/flux-3-image | Multi-reference generation and editing | 768, 1K, 1.5K, 2K, 4K |  | 2 |
| openai/gpt-image-2.5-sunburst | Precision-oriented generation and editing tier |  | auto, low, medium, high, xhigh, max | 2 |
| openai/gpt-image-2.5-flare | Speed-oriented generation and editing tier |  | auto, low, medium, high, xhigh, max | 2 |

Architecture details, parameter counts and downloadable weight sizes are not exposed by this endpoint inventory. No phone-deployment claim follows from a hosted API result.

Type/positioning labels above summarize the provider catalog descriptions; they are not independent quality claims. Actual reference-input support and controls come from endpoint capabilities.

An empty control column means the endpoint exposes no such control. It does not mean the model cannot internally change pixel count. Medium and Large in Krea model names are separate model variants, not selectable quality levels.

Recraft V4.1 Flash excluded: this endpoint accepts text only. Both Ming models, Meta Muse and the catalog page excluded at user request.

49 profiles × 2 covers = 98 comparison cells: 82 new requests, 11 reused successful baseline outputs, 5 FLUX road-cover cells withheld following its prior provider rejection. No filtered-cover retries.

Single-reference models receive only the centered green target canvas. Other models receive that canvas and the source square. The prompt is identical except the second-reference sentence is removed for single-reference models.

Pinned providers and prices are retained in [the plan](zeytin-plan.json). Actual charges come from returned usage and generation billing. Krea pricing is absent in its endpoint catalog; this is not treated as zero cost.

Sources: [OpenRouter Image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation), [OpenAI image controls](https://developers.openai.com/api/docs/guides/image-generation).
