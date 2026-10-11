# Nergis — active teacher model scope

2026-10-11. The user clarified that the active comparison and future teacher
experiments must include only:

- `openai/gpt-image-2.5-sunburst`
- `openai/gpt-image-2.5-flare`
- `google/gemini-nano-banana-2.1`

GPT Image 2 and all other legacy models are historical records, not active
candidates. Mersin previously defaulted to the entire preserved history; its
19-model total did not represent new generation in that stage.

The Mersin viewer now defaults to the selected three models, across every
existing prompt. It shows **106 request records and 88 existing images**:
Sunburst 41/39, Flare 39/36, Nano 26/13 (records/images). The model selector,
coverage matrix and output cards all use the same active scope. Direct Google
and OpenRouter Nano requests remain identifiable by route.

The complete 19-model, 176-record, 151-image history is available only through
the explicit historical scope selector. Historical outputs and their receipts
were preserved. The earlier liked Zeytin Sunburst reference remains visible.
No new provider request, generation, model download or generation charge
occurred. The existing saved archive JSON was reused to refresh viewer markup.

Browser verification confirmed the exact three active model IDs and 106/88
counts, the full historical 176/151 counts, and restoration of active scope.
Local review: http://127.0.0.1:54146/mersin/ .
