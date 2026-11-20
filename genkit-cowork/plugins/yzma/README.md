# Genkit Yzma plugin

Register local GGUF models with Genkit using
[Yzma](https://github.com/hybridgroup/yzma)'s in-process llama.cpp bindings.
No Ollama or vLLM server is required.

Supports metadata-only discovery, chat, raw text completion, and streaming for
decoder-only models. Weights load on demand, and each request gets a fresh
inference context and sampler. Tool calling requires an explicit definition
for a tool-capable chat model with a tool-aware template.
Chat and completion also support schema-constrained JSON objects without tools
 or streaming. Vision chat models accept inline JPEG and PNG images.
 Explicitly registered Soprano models synthesize speech from text without
accepting media input. System One models score typed decisions through Yzma's
local readouts.

## Installation

The plugin can be used independently of cowork's flows, tools, memory, and skills.
It is a package in the root Go module, not a separately versioned module.

1. Install Go 1.26 or newer.
2. Add the plugin package to your application. Replace `REVISION` with a pushed
   tag, branch, or commit containing this plugin. Releases predating the plugin
   cannot install it.

   ```sh
   go get github.com/TheSlowpes/genkit-cowork/genkit-cowork/plugins/yzma@REVISION
   ```

3. Install the Yzma CLI version matching this module's pinned dependency and use it
   to download compatible llama.cpp libraries. These commands use a Unix shell:

   ```sh
   go install github.com/hybridgroup/yzma@v1.33.0
   yzma install --lib "$HOME/.local/lib/yzma"
   export YZMA_LIB="$HOME/.local/lib/yzma"
   ```

   Ensure Go's binary directory is on `PATH`. Using the pinned CLI's default
   library release avoids mismatched native bindings. For platform and hardware
   options, see [Yzma installation](https://yzma.ai/getting-started/install/).
   A C compiler is not required, but the native shared libraries are.

4. Download a compatible GGUF model or use an existing local file. Model downloads
   are separate from library installation; the plugin does not download models.
   See [Yzma model downloads](https://yzma.ai/getting-started/download-models/).

## Use with Genkit

Pass `&yzma.Yzma{}` to `genkit.WithPlugins`. Genkit can then list installed
models in the Dev UI and resolve them by name, even before it has listed them.
Use `ModelsDir` to point at a different catalog. No other cowork components
need registration.

### Discover local models in the Dev UI

Install the [Genkit CLI](https://genkit.dev/docs/devtools/) and the compatible
native libraries described above. Put a local `SmolLM2-135M.Q4_K_M.gguf` in
`$HOME/models`. Use the base model, not the `Instruct` variant. The tested
105 MB artifact is
[`QuantFactory/SmolLM2-135M-GGUF`](https://huggingface.co/QuantFactory/SmolLM2-135M-GGUF/blob/main/SmolLM2-135M.Q4_K_M.gguf).

From the repository root:

```sh
export YZMA_LIB="$HOME/.local/lib/yzma"
task dev:yzma:ui -- -models-dir "$HOME/models"
```

1. Open the URL printed by Genkit. Under **Models**, select
   `yzma/smollm2-135m.q4_k_m`.
2. Send one user message, `Once upon a time`. Set `maxOutputTokens` to `16`
   in the model config to keep the run short. Expect streamed continuation
   text and a final response, not an echo of the prompt.
3. Press Ctrl+C to stop the example and free the weights.

No `DefineModel` or preload call is involved. If you already have
`gemma-4-E4B-it-Q4_0.gguf` with its canonical embedded chat template, select
`yzma/gemma-4-e4b-it-q4_0` for text chat. A projector is not required for
text; discovery does not enable media or tools for this model.

### What discovery includes

The plugin scans the top level of `ModelsDir`, or Yzma's default
`$HOME/models` if `ModelsDir` is empty. A missing default is an empty catalog.
A missing explicitly configured directory, a directory that cannot be read,
and skipped GGUFs produce warnings with the path and reason. The plugin never
downloads a file. Restart the app to refresh the catalog after moving, adding,
or replacing files. Avoid changing files in a running catalog.

IDs are `yzma/` plus the lowercased filename without `.gguf`. Use distinct
names for different quantizations. Names must start with an ASCII letter or
digit and otherwise contain only ASCII letters, digits, dots, hyphens, and
underscores. Discovery skips every file in an ID collision. A conventional
split model keeps its `-00001-of-NNNNN` suffix in the ID; only the first shard
can be selected and its matching sibling files must be present.

Discovery reads the GGUF header and tensor directory without loading weights.
For generation, it recognizes `llama` and `gemma4` GGUFs marked `general.type=model`
with a tokenizer and decoder tensors, and rejects pooling, non-causal
attention, and known classifier heads. Projectors, adapters, and unclassified
GGUFs stay out of the list. On a recognized decoder, a valid default embedded
template selects chat formatting; no template selects raw continuation. A
file with only named templates needs an explicit definition unless it has a
recognized [System One readout](#typed-system-one-decisions). BERT embedding
models follow the separate [embedding rules](#local-text-embeddings).
 SmolVLM discovery also checks for a [compatible projector](#ask-a-vision-model-about-an-image).

These are checks for a usable text path, not proof of training intent,
tool-calling support, native compatibility, or available memory. For example,
the tested SmolLM2 file has a `llama` decoder, tied token embeddings, and no
chat template; the absence of that template does not prove that a decoder is
a base model. The Gemma file identifies `gemma4` and has the canonical default
template. Generate once to verify a local artifact; use `DefineModel` to
specify how to run a file discovery did not classify.

### Explicit registration

Call `DefineModel` during setup when you want your own name, settings, or role.
To override a discovered name, define it before the first lookup or generation
with that name. Listing alone does not reserve it. Genkit caches resolved
actions and rejects replacing an already registered name.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/plugins/yzma"
)

func main() {
	ctx := context.Background()
	local := &yzma.Yzma{} // Reads the native library directory from YZMA_LIB.
	g := genkit.Init(ctx, genkit.WithPlugins(local))
	defer func() {
		if err := local.Close(ctx); err != nil {
			log.Print(err)
		}
	}()

	cpuLayers := int32(0)
	local.DefineModel(g, yzma.ModelDefinition{
		Name: "local-chat",
		Type: yzma.ModelTypeChat,
		Loading: yzma.ModelOptions{
			GGUFPath:   "/path/to/model.gguf",
			NGPULayers: &cpuLayers,
		},
	}, nil)

	if err := local.Preload(ctx, "local-chat"); err != nil {
		log.Print(err)
		return
	}
	maxTokens, temperature := 64, 0.0
	text, err := genkit.GenerateText(ctx, g,
		ai.WithModelName("yzma/local-chat"),
		ai.WithPrompt("Say hello."),
		ai.WithConfig(&yzma.GenerateContentConfig{
			MaxOutputTokens: &maxTokens,
			Temperature:     &temperature,
		}),
	)
	if err != nil {
		log.Print(err)
		return
	}
	fmt.Println(text)
}
```

`DefineModel` validates and snapshots configuration without loading weights.
`Preload` takes the unprefixed registration name. To preload a discovered
model, first resolve it with `yzma.Model` or a Genkit model lookup. Both
`yzma.Model(g, "local-chat")` and `yzma.IsDefinedModel(g, "local-chat")`
resolve names without loading weights.
Preloading is optional: the first generation call loads the model when necessary.
Once loaded, `Preload` returns without waiting for inference, including when
called from a streaming callback.

### Choose a model type

`ModelDefinition.Type` is required for manual definitions. Choose the
formatting that matches the model.

| Type | Input and formatting | Genkit capabilities |
| --- | --- | --- |
| `yzma.ModelTypeChat` | System, user, and assistant messages formatted with a chat template | Multi-turn and system role |
| `yzma.ModelTypeCompletion` | Exactly one user text message, tokenized directly | Single-turn text completion |
| `yzma.ModelTypeSpeech` | Exactly one user text message | Complete WAV output, no media input |

For a base model without a chat template, register it as a completion model:

```go
local.DefineModel(g, yzma.ModelDefinition{
	Name: "local-completion",
	Type: yzma.ModelTypeCompletion,
	Loading: yzma.ModelOptions{
		GGUFPath: "/path/to/base-model.gguf",
	},
}, nil)

text, err := genkit.GenerateText(ctx, g,
	ai.WithModelName("yzma/local-completion"),
	ai.WithPrompt("Once upon a time"),
)
```

Completion mode does not read or apply chat templates, add an assistant prefix, or
interpret chat-special-token spellings in the prompt. The tokenizer still applies
the model's native special-token policy, such as a required beginning token.
The response contains the continuation, not an echo of the prompt.

Completion requests reject system messages, assistant messages, and multiple
messages rather than silently flattening them. `ChatTemplate` must be empty for a
completion registration. Missing and unknown model types are registration errors.
Genkit capability metadata comes from the type and cannot be overridden through
`ModelOptions.Supports`. `Label`, `Stage`, and `Versions` remain customizable.

Concurrent calls for one registered model share a successful load; failed loads
can be retried. Registration errors panic, while loading errors are returned by
`Preload` or generation. Each plugin instance allows one load at a time across
manual and discovered models. Loaded weights stay resident until `Close`.
Different quantizations or aliases can hold separate copies of weights; the
load limit does not cap resident memory.

Call `Close(ctx)` to reject new work, wait for active model use, and free weights.
If cleanup times out or fails, call `Close` again. Shared native libraries and
backends stay initialized for the process lifetime. All default-operator plugin
instances must use the same library directory. Do not independently unload
Yzma's native runtime while the plugin is in use.

## Configuration

| Configuration | Scope |
| --- | --- |
| `Yzma.LibPath` | Native library directory; falls back to `YZMA_LIB` |
| `Yzma.ModelsDir` | Catalog directory; falls back to Yzma's default |
| `Yzma.CatalogOperator` | Replace directory, GGUF header, and sidecar I/O for discovery |
| `Yzma.VideoDecoder` | Replace MP4 decoding for the built-in media operator |
| `ModelDefinition.Type` | Required chat, completion, speech, or decision mode |
| `ModelDefinition.Decision` | Decision readout and calibration sidecar path |
| `ModelDefinition.ToolCalling` | Opt into tool calling for a capable chat model |
| `ModelDefinition.ChatTemplateName` | Select a named template embedded in the GGUF |
| `ModelDefinition.Loading` | GGUF path, GPU placement, and weight-loading modes |
| `ModelDefinition.Context` | Token capacity, batch sizes, and CPU threads |
| `ModelDefinition.GenerationDefaults` | Sampling and output defaults |
| `GenerateContentConfig` | Per-request sampling and output overrides |
| `ModelDefinition.SpeechDecoderPath` | Required companion mmproj GGUF for speech |
| `SpeechConfig` | Per-request voice for speech |

Context and generation settings are applied on each request. `Verbose` is
reserved for future native logging configuration.

Optional numeric fields use pointers so an explicit zero differs from an omitted
value. Temperature zero requests greedy sampling; GPU layers zero keeps layers
on the CPU. Nil `StopSequences` inherits defaults, while a pointer to an empty
slice clears them.

Internal resolvers reject invalid modes and values. Loading and context settings
overlay llama.cpp's native defaults, which require an initialized native runtime.
`withModelParams` binds device and tensor-split arrays only for its synchronous
callback and keeps their storage alive until it returns. Do not retain the
callback's native pointers.

Generation starts with fresh Yzma sampler defaults, then applies model defaults
and request overrides. Output defaults to 1024 tokens. See
[`ExampleModelDefinition`](config_test.go) for configuration and
[`Example_resolveGenerationConfig`](resolve_test.go) for a greedy request
overriding the model's temperature without changing its output limit.

Set `Yzma.Operator` to a custom `ModelOperator` to replace native initialization,
loading, and cleanup for testing or alternate execution environments. To support
chat, that operator must also implement `TextGenerationOperator`.
`Yzma.CatalogOperator` is independent of model loading. It supplies directory
entries and file readers. The plugin still parses and validates bounded GGUF
headers and decision sidecars.

In chat mode, formatting reads the model's embedded template through llama.cpp,
with a GGUF metadata fallback. Full Jinja templates use the same Go Jinja engine
as Yzma's template package, with BOS/EOS variables and typed tool context supplied
by the plugin. This supports embedded templates that the native
`llama.ChatApplyTemplate` formatter cannot render, including Gemma 4's canonical
template. System, user, and prior assistant messages retain their roles.

Set `ModelDefinition.ChatTemplate` to an explicit Jinja source or a Yzma built-in
template name. `ChatTemplateName` selects a named embedded template instead.
With tools offered, an embedded `tool_use` template is preferred when available;
otherwise the default embedded template is used. Missing explicit names and
templates that ignore offered tool definitions return errors, not silent fallbacks.
Completion mode does not read templates.

Prompt tokens are evaluated in batches. Prompts that leave no room for output are
rejected, not truncated. Generation stops at an end-of-generation token, a stop
sequence, the output token limit, or the remaining context capacity. Stop text is
removed from both the stream and final response, including sequences spanning
token boundaries. Stream chunks contain complete UTF-8 text.

Responses include token usage and a `stop` or `length` finish reason. Output token
usage includes sampled end-of-generation tokens and stripped stop text.

## Tool calling

Enable `ToolCalling` only for a chat model trained to follow tool schemas, with a
template that formats tool definitions and tool results:

```go
local.DefineModel(g, yzma.ModelDefinition{
	Name: "local-chat",
	Type: yzma.ModelTypeChat,
	ToolCalling: true,
	Loading: yzma.ModelOptions{GGUFPath: "/path/to/tool-capable-model.gguf"},
}, nil)
```

Pass registered Genkit tools through `ai.WithTools(...)`. Genkit performs the
execution loop; the plugin never directly runs a model-requested function.
Use `auto` for automatic tool chains that should end with a text answer.
Genkit v1.9 forwards `required` on every turn, so the adapter requires a call
even after Genkit has executed a tool. To require a call only on the first
turn, use `required` with `ai.WithReturnToolRequests(true)` and make a separate
`auto` request after handling the tool result.

Tool-enabled turns are buffered until parsing completes. Streaming callbacks
receive complete text/tool parts, not raw tool-call syntax. Ordinary chat without
`ToolCalling` continues streaming text incrementally.

The adapter uses the same frames to read arguments and remove call syntax from
the answer. Prose outside those frames stays intact apart from outer whitespace
and recognized protocol markers. JSON objects and arrays in prose are opaque
unless an outer object has both `name` and `args` or `arguments`, in either
order. A call-shaped object inside another object or array is still prose.
A balanced outer object with both call keys is still a candidate when either
key is unquoted. Strict JSON parsing rejects it rather than repairing it.
An unclosed non-call JSON-shaped span does not hide a later explicit call
marker outside a quoted string.
Outside a Harmony text frame, Markdown fences do not change call detection.
Explicit thinking frames and prompt-injected thinking prefixes ending in
`</think>` or `<channel|>` never produce calls. An unclosed thinking frame, or
a malformed call inside one whose end cannot be located, hides the remainder.
Text after an end-of-turn marker is ignored, including simulated calls.

| Format | Accepted call framing |
| --- | --- |
| JSON / Phi | `<tool_call>{"name":"echo","args":{"text":"hi"}}</tool_call>`, also with the Phi opener `<\|tool_call>`, or as an inline object. Exactly one of `args` and `arguments` must be an object. |
| Qwen | `<function=echo><parameter=text>hi</parameter></function>`, bare or inside `<tool_call>…</tool_call>`. Inline spaces are literal; one matching pair of edge newlines is treated as template formatting. |
| Gemma | `call:echo{text:<\|"\|>hi<\|"\|>,items:[1,2]}`, bare or inside `<\|tool_call>…<tool_call\|>`. Supports nested objects, arrays, JSON strings, and the Gemma quote tokens `<\|"\|>`, `<">`, and `<\|>`. |
| GLM | `echo<arg_key>text</arg_key> <arg_value>hi</arg_value>`, starting on its own line. Spaces and tabs are allowed between a key and its value tag, but not a line break. The value body is literal and may contain line breaks. |
| LFM2 | `<\|tool_call_start\|>…<\|tool_call_end\|>` around a JSON call object, a nonempty list of those objects, or a nonempty Python-style list such as `[echo(text='hi')]`. |
| Mistral / Devstral | `[TOOL_CALLS]echo[ARGS]{"text":"hi"}`, optionally followed by `[/TOOL_CALLS]`. Repeat the entire frame for each call. |
| Harmony | Commentary calls addressed to `functions.*` with a JSON object and a closing `<\|call\|>`. The recipient may be in the role or commentary header. The older `.echo <\|message\|>{"text":"hi"}` shorthand ends at its JSON object. Analysis is hidden. Final message bodies are visible text, even when they contain call examples. |

The whole batch returns an error before dispatch if any visible candidate is
malformed, has a missing closer, contains duplicate keys (including nested
objects), names an unoffered tool, or fails its input schema. Recognizable
truncated prefixes also fail, including inline JSON that starts with an argument
key but never finishes. Batch members cannot mix formats. Wrapped, inline, and
Phi JSON count as one format; LFM2 JSON and Python count separately.
A turn that hits its output or context limit does not dispatch even a
complete-looking call.

The parser does not repair older Mistral `[TOOL_CALLS][…]` arrays, alternate
`<toolcall>` wrappers, bare Gemma strings or missing commas, or Qwen/GLM values
with unquoted occurrences of their own delimiters. Those delimiters can appear
inside a leading JSON value in a Qwen or GLM argument, unless a value's closing
tag is followed by another argument opener or, in Qwen, a function opener,
function closer, or wrapper closer. A Gemma quote token cannot appear inside
a string delimited by that same token; use JSON quotes instead.

LFM2 Python accepts keyword arguments, strings, lists, dictionaries
with string keys, numbers in JSON notation, and Python or JSON spellings of
booleans and null. Strings accept quote and backslash escapes, `\a\b\f\n\r\t\v`,
and `\x`, `\u`, and `\U` escapes. JSON-style double-quoted strings also accept
paired UTF-16 escapes. Positional arguments, expressions, tuples,
string prefixes, and other escapes are rejected rather than reinterpreted.

Tool inputs must use JSON object schemas. Local schema references are expanded
for rendering; recursive schemas are rejected. Qwen and GLM argument tags
provide text, so the offered schema must give each parameter an unambiguous
type to interpret it. Typed literals keep their types. Numbers that would lose
precision and explicit null object fields that Genkit would drop return errors.
Omit optional object fields instead. Arrays may contain null if allowed by the
schema. Call IDs stay with the arguments and tool results in later turns.

The opt-in native test runs a real Genkit tool with a local Gemma 4 model. It
checks that Genkit executes it once, correlates the result with the call ID, and
feeds a secret value back to the model.

```sh
YZMA_LIB="$HOME/.local/lib" \
YZMA_TEST_TOOL_MODEL="$HOME/models/gemma-4-E4B-it-Q4_0.gguf" \
  task test:yzma:tools:native
```

For example, if `text` is a string, GLM's
`<arg_key>text</arg_key> <arg_value>{"name":"Ada"}</arg_value>` passes the
original JSON-shaped text, not a new call or a decoded object. Use string
fields for IDs such as `9007199254740993` that would change when Genkit v1.9
decodes numbers through `float64`.

## Constrained structured output

Use `genkit.GenerateData[T]` with a JSON-tagged Go struct. The plugin derives
a llama.cpp grammar from the Genkit schema, applies it before token selection,
and validates the raw output against the schema before Genkit decodes it into
`T`. A prompt can still explain what the fields mean: Genkit does not add
schema instructions in native constrained mode. A schema constrains shape and
allowed values, not whether the model's answer is factually correct.
You can use a discovered model name or an explicit definition.

The supported JSON Schema subset is:

| Schema type | Supported keywords and rules |
| --- | --- |
| `object` | `properties`, `required`, `additionalProperties: false`. The root must be a closed object. Nested objects are also closed. The grammar writes properties in sorted order. |
| `array` | `items` with a single schema from this subset. |
| `string` | An optional `enum` of distinct strings. Ordinary strings use literal UTF-8 and standard short JSON escapes. |
| `integer`, `number` | No extra keywords. To fit Genkit v1.9's float64 intermediate representation, the grammar allows up to 15 digits before the decimal point and up to two exponent digits. |
| `boolean`, `null` | No extra keywords. |

Every schema node must have one `type`. `title` and `description` are accepted
as annotations. The plugin rejects any other keyword, including references,
compositions, string patterns, size bounds, and numeric bounds. Schemas are
limited to 64 KiB, 256 nodes, and 24 levels.

Do not pass tools, tool history, streaming callbacks, or stop sequences to a
constrained request. This includes tools supplied with `ToolChoiceNone`.
Token or context exhaustion returns an error, even if a closing brace was
already emitted. Increase the limit to allow the end-of-generation token.
Genkit v1.9 can remove the schema and constraint flag when tools are supplied;
the plugin rejects the remaining JSON request instead of using prompt-only
generation. Leave the default JSON formatter in place when using
`GenerateData[T]`. A formatter that discards all structured output settings
before Genkit calls the model cannot be distinguished from a plain text request.
Custom `ModelOperator` implementations must also implement
`ConstrainedTextGenerationOperator` to advertise and run this feature.

To run the [structured output example](../../../examples/yzma-structured/main.go),
install the native libraries described above and use the base
`SmolLM2-135M.Q4_K_M.gguf` in completion mode. From the repository root:

```sh
YZMA_LIB="$HOME/.local/lib/yzma" task dev:yzma:structured -- \
  -model "$HOME/models/SmolLM2-135M.Q4_K_M.gguf"
```

Expected output: `Decoded Go value: {Code:YZMA}`. The prompt asks for plain text,
but the schema permits only a JSON object with the `code` value `"YZMA"`.
The example uses the CPU and does not download model files.

## Speech output

Register a Soprano backbone with `Type: yzma.ModelTypeSpeech` and set
`SpeechDecoderPath` to its matching `mmproj` GGUF. The companion holds the decoder
that converts model output to audio. Speech pairs require explicit registration;
text discovery does not select the decoder. The backbone and decoder load
together on first use or `Preload`. Both stay resident until `Close`. Speech
uses Yzma v1.33.0's in-process `libmtmd` audio generation helpers, verified on
llama.cpp `b12359`. No `llama-tts` executable or media input is needed.

Call `genkit.Generate` with one user message of plain English text. Soprano uses
its fixed voice when `SpeechConfig.Voice` is omitted or set to `"default"`. Any
other name returns an error, as do audio input, tools, and streaming. The prompt
must fit in 1024 UTF-8 bytes. If generation reaches its 512-frame limit or runs
out of context before the end of speech, the plugin returns an error instead of
partial audio.

The response has one `audio/wav` media part and a `stop` finish reason.
`res.Media()` returns `data:audio/wav;base64,<payload>`. Remove that prefix and
base64-decode the payload to get a mono, little-endian, signed 16-bit PCM WAV at
32,000 Hz. Do not set a Genkit output formatter; Genkit v1.9 has no built-in
media formatter, and the registered model already selects the output format.

Run the [speech example](../../../examples/yzma-speech/main.go) with these two
public files from [ggml-org/Soprano-1.1-80M-GGUF](https://huggingface.co/ggml-org/Soprano-1.1-80M-GGUF):

| Artifact | Download size |
| --- | ---: |
| [Soprano-1.1-80M-Q8_0.gguf](https://huggingface.co/ggml-org/Soprano-1.1-80M-GGUF/resolve/main/Soprano-1.1-80M-Q8_0.gguf) | 84.9 MB |
| [mmproj-Soprano-1.1-80M-F16.gguf](https://huggingface.co/ggml-org/Soprano-1.1-80M-GGUF/resolve/main/mmproj-Soprano-1.1-80M-F16.gguf) | 60.8 MB |

Save both files under `$HOME/models/soprano`. From the repository root, point
`YZMA_LIB` at your Yzma library installation and run:

```sh
export YZMA_LIB="$HOME/.local/lib/yzma"
SPEECH_MODELS="$HOME/models/soprano"
task dev:yzma:speech -- \
  -model "$SPEECH_MODELS/Soprano-1.1-80M-Q8_0.gguf" \
  -decoder "$SPEECH_MODELS/mmproj-Soprano-1.1-80M-F16.gguf" \
  -voice default -text "Hello from Genkit." -out speech.wav
ffplay -autoexit -nodisp speech.wav
```

The example runs on CPU and writes a short clip that says "Hello from Genkit."
You can also open `speech.wav` in any WAV player. To check the same pair through
Genkit without writing a file:

```sh
YZMA_TEST_SPEECH_MODEL="$SPEECH_MODELS/Soprano-1.1-80M-Q8_0.gguf" \
YZMA_TEST_SPEECH_DECODER="$SPEECH_MODELS/mmproj-Soprano-1.1-80M-F16.gguf" \
  task test:yzma:speech:native
```

The native test uses existing local files. Unit tests and CI do not need the
model or `libmtmd`. A custom `ModelOperator` must implement `SpeechOperator`
to register a speech model.

## Typed System One decisions

Use `genkit.GenerateData[T]` with a struct of decision questions. Each top-level
field has a `json` key and a `jsonschema_description` question. The plugin
extracts the questions from the output schema and scores them in one local
`DecideMany` call. It follows the
[Genkit System One API pattern](https://genkit.dev/docs/go/integrations/system-one/)
without using the remote plugin or changing the Genkit version.

| Field type | What it returns |
| --- | --- |
| `yzma.Choice[T]` | The winning option, probabilities keyed by option, and readout confidence. `T` is a string type with `Criteria() map[T]string`. Options are sent in key order. |
| `yzma.Score[L]` | The probability-weighted mean level, a probability per level, readout confidence, and the rubric legend. `L` has `Levels() []string` with two to ten levels, lowest first. |
| `yzma.Noul` | The probability of yes. The probability of no is `1 - Probability`. |

Use one user message as state. Pass plain text with `ai.WithPrompt`, or use
`ai.WithMessages(ai.NewUserMessage(ai.NewDataPart(rawJSON)))` for a JSON object
or array encoded as a string. Optional `ai.WithSystem` text prefixes each
question's instructions rather than becoming part of the state. Nested
questions, extra turns, attachments, media, tools, text completion, streaming,
and per-request sampling settings are not supported. Model and native-library
limits still apply. Confidence is readout-specific and is not interchangeable
with an option probability. Retune thresholds when changing the model,
quantization, or calibration.

### Readouts and calibration files

The catalog can recognize these artifacts without loading their weights:

| `Decision.Readout` | How discovery identifies it |
| --- | --- |
| `yzma.DecisionReadoutGGUF` | Recognized `<architecture>.decision.type` plus the `tokenizer.chat_template.systemone` template. Calibration is embedded. Supported metadata types follow Yzma v1.33.0's `decide.Open`. |
| `yzma.DecisionReadoutJevStyle` | `<GGUF stem>.readout.json`, the model's `readout_config.json`. |
| `yzma.DecisionReadoutJevK5` | `<GGUF stem>.jevk5.json`, the model's `jevk5_config.json`. |
| `yzma.DecisionReadoutDecider` | `<GGUF stem>.decider.json`, the model's `decider_config.json`. Use the plain layout. |

Match the case of the GGUF filename stem when naming the sidecar. Use the
calibration file published for that model version, not a file from another
release. Discovery skips malformed or conflicting readouts. A generic sidecar
name such as `decider_config.json` is not enough for discovery to identify its
GGUF. To pair arbitrary paths or override an ambiguous readout, register it
before resolving the model:

```go
local.DefineModel(g, yzma.ModelDefinition{
    Name: "support-decider",
    Type: yzma.ModelTypeDecision,
    Loading: yzma.ModelOptions{GGUFPath: "/path/to/model.gguf"},
    Decision: yzma.DecisionOptions{
        Readout:    yzma.DecisionReadoutDecider,
        ConfigPath: "/path/to/decider_config.json",
    },
}, nil)
```

For an embedded readout, use `DecisionReadoutGGUF` without `ConfigPath`.
Decision models use Yzma's native weight-loading defaults. An explicit
definition can set `ContextSize`, `PhysicalMaximumBatchSize`, and `Threads`
through `ModelDefinition.Context`. The decision reader keeps both weights and
its inference context until `Close` and may reuse a shared state prefix.
Calls on one reader are serialized. Cancellation stops waits for the model and
discards a canceled decision, but pinned Yzma cannot interrupt a native decision
load or score already in progress. `Close` waits for active scoring.

### Run the decision example

Use the public Decider 0.8B Q2_K GGUF (422 MB) and its 341-byte calibration file.
With the compatible native libraries from [installation](#installation), run
from the repository root:

```sh
export YZMA_LIB="$HOME/.local/lib/yzma"
dir="${TMPDIR:-/tmp}/yzma-decision-models"
mkdir -p "$dir"
curl -fL -o "$dir/decider-0.8b.Q2_K.gguf" \
  https://huggingface.co/mradermacher/decider-0.8b-GGUF/resolve/001df41ddc126176cab96c8aec9ada45008fbe62/decider-0.8b.Q2_K.gguf
curl -fL -o "$dir/decider-0.8b.Q2_K.decider.json" \
  https://huggingface.co/Mapika/decider-0.8b/resolve/a0a01d6f8135298f400a8c856b355793012ae971/decider_config.json
task dev:yzma:decision -- -models-dir "$dir" -model decider-0.8b.q2_k
```

The [example](../../../examples/yzma-decision/main.go) prints the typed result
as JSON. For the default ticket, expect `department.choice` to be `billing`,
`department.probabilities` to include all three options, and `urgent.probability`
to be above `0.5`.

Use the same files for the opt-in native test:

```sh
YZMA_TEST_DECISION_MODEL="$dir/decider-0.8b.Q2_K.gguf" task test:yzma:decision:native
```

## Run an explicit definition in the Genkit Dev UI

Install the [Genkit CLI](https://genkit.dev/docs/devtools/) if needed, then run
these commands from the repository root:

```sh
export YZMA_LIB=/path/to/libs
task dev:yzma:ui -- -model /path/to/model.gguf
```

Open the Dev UI at the address printed by Genkit. Under **Models**, select
`yzma/local-chat` and send a message. The example preloads the weights, keeps the
runtime alive, and streams responses. Press Ctrl+C to stop and free the weights.
It uses CPU-only placement and a 4096-token context.

For a base model, use `-type completion` and select `yzma/local-completion` under
Models. Supply a single text prompt; this mode does not support a multi-turn chat:

```sh
task dev:yzma:ui -- -model ~/models/SmolLM2-135M.Q4_K_M.gguf -type completion
```

The example uses embedded Jinja templates automatically, including the template
in `gemma-4-E4B-it-Q4_0.gguf`. `-template-name NAME` selects a named GGUF template.
`-template NAME` can select a Yzma built-in override matching the model's trained
format; the plugin does not guess a replacement.

To test tools with your Gemma 4 model:

```sh
task dev:yzma:ui -- -model ~/models/gemma-4-E4B-it-Q4_0.gguf -tools
```

Select **Flows → localChat**, and use `{"prompt":"Use add to calculate 2 + 3."}`.
The trace shows Genkit executing the registered `add` tool and sending its result
back to the model. The model's instruction-following quality still determines
whether an `auto` request chooses the tool.

## Run the example

From the repository root, load and release a model on the CPU without inference:

```sh
YZMA_LIB=/path/to/libs task dev:yzma -- -model /path/to/model.gguf
```

The [example](../../../examples/yzma-loading/main.go) handles cancellation and
cleans up with a separate timeout.

To generate a single response with a 256-token limit, add `-prompt`.
`-max-tokens` changes the per-call limit:

```sh
YZMA_LIB=/path/to/libs task dev:yzma -- \
  -model /path/to/model.gguf -prompt "Say hello."
```

To continue a raw prompt with a base model:

```sh
YZMA_LIB=/path/to/libs task dev:yzma -- \
  -model ~/models/SmolLM2-135M.Q4_K_M.gguf \
  -type completion -prompt "Once upon a time"
```

## Tests

Run from the repository root. Unit tests use a custom `ModelOperator` and require
no native installation:

```sh
task test:yzma:race
```

To exercise loading, streaming generation, and the embedded template with
existing native libraries and a local model:

```sh
YZMA_LIB=/path/to/libs YZMA_TEST_MODEL=/path/to/model.gguf task test:yzma:native
```

The native task requires both paths and does not download models. `YZMA_TEST_TYPE=completion`
selects raw completion for a base model; the default is chat. `YZMA_TEST_TEMPLATE`
optionally overrides the chat template. The embedded template test skips models
without an embedded chat template. The discovery test uses the file without
overrides. To run that test on its own against the example's base model:

```sh
YZMA_LIB=/path/to/libs YZMA_TEST_MODEL=~/models/SmolLM2-135M.Q4_K_M.gguf \
  task test:yzma:native -- -run '^TestNativeDiscoveredTextGeneration$'
```

To verify constrained generation with the base SmolLM2 model:

```sh
YZMA_LIB="$HOME/.local/lib/yzma" YZMA_TEST_MODEL="$HOME/models/SmolLM2-135M.Q4_K_M.gguf" \
  task test:yzma:native -- -run '^TestNativeStructuredOutput$'
```

To test the embedded template without allocating model weights:

```sh
YZMA_LIB=/path/to/libs YZMA_TEST_MODEL=~/models/gemma-4-E4B-it-Q4_0.gguf \
  task test:yzma:native -- -run '^TestNativeEmbeddedTemplate$'
```

## Local text embeddings

Use `genkit.Embed` with a `yzma/<filename-stem>` embedder name. Discovery
lists a GGUF as an embedder when its header identifies a non-causal `bert`
encoder with a BERT tokenizer, embedding dimensions, encoder tensors, and
CLS or mean pooling. It does not list ranking, unpooled, or unclassified
files as embedders. Listing and lookup read no weights. Use
`genkit.LookupEmbedder` with the qualified name to resolve an embedder
before calling `Preload` with the unprefixed name, if preloading is needed.

An embed request returns one vector per input document in input order.
Text parts in a document are joined without separators. The embedder rejects
empty documents, non-text parts, and per-request options. It does not add
a query or passage prefix; add any prefix your model expects to the text.
The tokenizer applies the GGUF's special-token policy.

Discovered models use the pooling declared by their GGUF. CLS pooling uses
the first token's representation, while mean pooling averages the sequence's
token representations. The plugin L2-normalizes each vector and rejects
non-finite or zero-norm output. The vector length is fixed for the model, is
advertised in the embedder descriptor, and must match the native output size.
There is no dimension override per request.

Each document must fit the trained context, configured context, logical
batch, and physical microbatch limits. A pooled encoder must see the
whole document at once. An over-limit document fails the whole request;
the plugin does not truncate it or return partial results. The default
embedding context uses the GGUF's trained sequence length, subject to
the native batch limits. The tested BGE file supports up to 512 tokens
including special tokens.

To override a catalog name, or to register a BERT GGUF with
insufficient role metadata, call `DefineEmbedder` before the first lookup:

```go
local.DefineEmbedder(g, yzma.EmbeddingDefinition{
    Name:       "local-bge",
    Loading:    yzma.ModelOptions{GGUFPath: "/path/to/bge-small-en-v1.5-q4_k_m.gguf"},
    Pooling:    llama.PoolingTypeCLS,
    Dimensions: 384,
})
```

`llama` is `github.com/hybridgroup/yzma/pkg/llama`. Choose the pooling
and output dimensions the model was trained for. An explicit definition can
also set `Context` and loading options. Embedders share the plugin's load
limit and `Close` lifecycle with generation models. A custom `ModelOperator`
must also implement `TextEmbeddingOperator` to run embedding requests.

### Run the embedding example

Install the native libraries described above. Use
[`CompendiumLabs/bge-small-en-v1.5-gguf`'s `bge-small-en-v1.5-q4_k_m.gguf`](https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/blob/main/bge-small-en-v1.5-q4_k_m.gguf)
(24,808,576 bytes, SHA-256 `363a0a4855dff6c653e06efe3209157debcf7f74e52d0d7c71e2747cd523043e`).
Place it in `$HOME/models` under that filename. From the repository root:

```sh
YZMA_LIB="$HOME/.local/lib/yzma" go run ./examples/yzma-embeddings \
  -models-dir "$HOME/models"
```

The [example](../../../examples/yzma-embeddings/main.go) does not download
files. After any native log messages, expect:

```text
Embedding count: 2
Vector 1: 384 dimensions
Vector 2: 384 dimensions
```

Run the opt-in native test with the same fixture. It compares ordered batch
results to individual encodes, checks the vector norms, and rejects context
overflow:

```sh
YZMA_LIB="$HOME/.local/lib/yzma" YZMA_TEST_TYPE=embedding \
  YZMA_TEST_MODEL="$HOME/models/bge-small-en-v1.5-q4_k_m.gguf" \
  task test:yzma:native -- -run '^TestNativeDiscoveredEmbedding$'
```

## Ask a vision model about an image

Use the public [SmolVLM 256M Instruct GGUF pair](https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF):
`SmolVLM-256M-Instruct-Q8_0.gguf` (175 MB) and
`mmproj-SmolVLM-256M-Instruct-Q8_0.gguf` (104 MB). Put both files in
`$HOME/models`. The library directory in `YZMA_LIB` must contain `mtmd`
alongside the llama.cpp libraries. From the repository root:

```sh
export YZMA_LIB="$HOME/.local/lib/yzma"
task setup
task dev:yzma:vision -- \
  -model "$HOME/models/SmolVLM-256M-Instruct-Q8_0.gguf" \
  -projector "$HOME/models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf" \
  -image "$(go env GOMODCACHE)/github.com/hybridgroup/yzma@v1.33.0/images/domestic_llama.jpg"
```

The program reads the photo and sends it inline between text parts. Expect an
`Answer:` line describing the animal. In the tested run, the model described a
woolly animal standing in a grassy area. It guessed alpaca or yak, although the
source photo is labeled as a llama. Use `-question` to ask a different question.

Discovery recognizes SmolVLM decoders and matches the projector by GGUF model
name, projector type and projection width. When several compatible projectors
are installed, `mmproj-<decoder filename>` selects the same quantization.
Otherwise, discovery requires exactly one compatible projector. If none match
or the match is ambiguous, register the decoder with `DefineModel` and set
`Projector: &yzma.ProjectorOptions{GGUFPath: path, Images: true}`. Projectors are
never standalone Genkit actions. Other vision families require explicit pairing
and a compatible chat template that accepts string message content. Only the
SmolVLM pair above is verified here.

- Send `ai.NewMediaPart("image/png", "data:image/png;base64,...")` or the JPEG
  equivalent in a user message. The plugin does not fetch URLs or open paths.
- The limits per request are four images, 4 MiB per image, and 8 MiB total
  before base64 encoding. Width and height must each be at most 2048 pixels.
  The resulting image and text tokens must also fit the model's context.
- Vision requests can stream text or use [constrained JSON](#constrained-structured-output)
  without streaming. Neither mode accepts tools. Constrained media requests
  accept only images.
  Image support alone does not enable [video input](#ask-a-vision-model-about-a-video).
  After a successful load, the model and projector stay
  resident until `Close`, which frees the projector before its model. If
  loading a projector or speech decoder fails or is canceled, the plugin
  rolls back the pair. If cleanup fails, the next load or `Close` retries it
  before the model can be used. Use a fresh cleanup
  context if the request or application context has been canceled.

The native test sends two colored images in both orders through Genkit and
checks that the model identifies the first image:

```sh
YZMA_TEST_MODEL="$HOME/models/SmolVLM-256M-Instruct-Q8_0.gguf" \
YZMA_TEST_PROJECTOR="$HOME/models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf" \
  task test:yzma:native -- -run '^TestNativeVisionGeneration$'
```

## Predict a bounding box

Call `yzma.GenerateBoundingBox(ctx, g, model, image, instruction)` with a
Yzma vision model, an inline JPEG or PNG Genkit media part, and an instruction
such as `"Locate the llama in the image."` The helper uses
`genkit.GenerateData[yzma.BoundingBox]` with native grammar sampling, then
checks the box geometry. Custom vision operators must implement
`ConstrainedMediaGenerationOperator` to advertise and run this path.

The result is one box with `left`, `top`, `right`, and `bottom` integer edges.
The origin is the top-left corner. X increases rightward and Y downward.
Coordinates are in thousandths of the image width or height, so `1000` is
the far edge. Use `(left, top)` for the minimum corner and `(right, bottom)`
for the maximum corner. For an overlay, multiply each horizontal edge by
`width / 1000.0` and each vertical edge by `height / 1000.0`.

`GenerateBoundingBox` returns an error for coordinates outside `[0, 1000]`
or any box where `left >= right` or `top >= bottom`. It never clamps or
reorders coordinates. A valid box can still be wrong about what is in the
image. The result is data only; this call supplies no tools or action
execution.

The [Yzma model guide](https://yzma.ai/docs/guides/models/#pelican10-vl-3b)
lists Pelican1.0-VL-3B for grounding and shows its `bounding_box_2d` array
output. To run the check, use this
[pinned Pelican GGUF pair](https://huggingface.co/mradermacher/Pelican1.0-VL-3B-GGUF/tree/b08afd4ebda9573bc6b1f7242a52a3e30ff44b40):

| File | Bytes |
| --- | ---: |
| `Pelican1.0-VL-3B.Q4_K_M.gguf` | 1,929,901,568 |
| `Pelican1.0-VL-3B.mmproj-Q8_0.gguf` | 847,768,384 |

The fixture and test do not download the files. The example uses
`ProjectorOptions.ImageMinTokens: 1024`, the minimum llama.cpp recommends
for Qwen-VL grounding. This increases the encoder resolution without
changing the fixture or how the boxes are scored. A zero value leaves the
native default unchanged for other models.

`-check-grounding` generates two images with the same red target in different
positions, plus a blue distractor. It asks the same question for both images
and requires at least 0.5 intersection-over-union with each target. A whole-frame
box scores 0.09 and fails. The example checks both images and exits nonzero
if either misses.

```sh
export YZMA_LIB="$HOME/.local/lib"
export YZMA_TEST_VLA_MODEL="$HOME/models/Pelican1.0-VL-3B.Q4_K_M.gguf"
export YZMA_TEST_VLA_PROJECTOR="$HOME/models/Pelican1.0-VL-3B.mmproj-Q8_0.gguf"
task dev:yzma:vision -- -check-grounding \
  -model "$YZMA_TEST_VLA_MODEL" -projector "$YZMA_TEST_VLA_PROJECTOR"
```

The opt-in native test uses the same fixtures and acceptance check:

```sh
task test:yzma:grounding:native
```

**Observed CPU result:** With Yzma v1.33.0 and llama.cpp build 12359
(`b1b770d90`), both images loaded and each encoded to 1024 vision tokens.
Both model predictions were valid boxes, but neither met the 0.5 IoU gate:

| Red square | Expected `(left, top, right, bottom)` | Predicted | IoU |
| --- | --- | --- | ---: |
| Left | `(100, 200, 400, 500)` | `(150, 200, 450, 340)` | 0.361 |
| Right | `(600, 200, 900, 500)` | `(600, 300, 800, 500)` | 0.444 |

Without the projector override, each image used 121 vision tokens and IoU
was 0.000 and 0.156, respectively. A stock llama.cpp multimodal CLI control
using `bounding_box_2d`, normalized coordinates, and 1024 tokens returned
`[178, 203, 359, 448]` for the left image (0.493 IoU) and a whole-frame box
for the right image (0.090 IoU). The array representation did not meet the
gate either. These runs do not verify grounding on both targets.

An earlier CPU run with Qwen3.5-0.8B Q4_0 and its F16 projector returned
zero-width boxes for both fixtures. It confirmed image input, not grounding.

Use `-box -image path.png -question '…'` for your own image. That mode checks
geometry only because there is no known answer to compare with.

## Ask a vision model about a video

Register the [SmolVLM 256M Instruct GGUF pair](#ask-a-vision-model-about-an-image)
with `Projector: &yzma.ProjectorOptions{GGUFPath: path, Video: true}`. Video
requires explicit opt-in and a matching idefics3 projector. You can also
set `Images: true` to accept standalone images in the same request.
Install `ffmpeg` and `ffprobe` on `PATH` for video requests. The tested
version is FFmpeg 6.1. Other model families and native video modes are not
supported by this path.

Pass `ai.NewMediaPart("video/mp4", "data:video/mp4;base64,...")` in a user
message. The plugin accepts non-fragmented, fast-start MP4 with H.264 in
the first video track and explicit 32-bit box sizes. It rejects local
paths, remote URLs, malformed clips, and unknown duration or frame counts.
Limits before base64
encoding: 4 MiB per clip, four media parts and 8 MiB of media per request.
Both container and video track must be at most 10 seconds, with 1–300 frames
in the video track and declared dimensions of at most 1024 pixels per edge.

The plugin checks the decoded frame count against the declared count and
rejects video samples outside the allowed timeline. Probing and decoding
share a 15-second timeout per clip. The plugin keeps up to four frames
spaced by display-order **frame index**, including the first and last: for
`n` frames and `k > 1` samples, keep index `floor(i*(n-1)/(k-1))` for
`i` from `0` through `k-1`. Variable-rate clips are not sampled at equal
time intervals. Frames are resized with aspect ratio preserved and padded
to 256 × 256 pixels, then passed with `Frame 1:`, `Frame 2:`, and so on at
the clip's position in the prompt. No audio track is decoded or passed to the model.
Other audio, subtitle, and data tracks do not contribute to the prompt.
All text and sampled frames must fit the model's context window.
Video requests can stream text but cannot use tools or constrained output.

The Genkit adapter validates the MP4 container and request budgets before
passing `MediaGenerationInput` to the operator. The built-in operator probes
and samples the clip before loading model weights. A custom
`MediaGenerationOperator` gets the encoded clip bytes and no decoded frames.
It owns track validation and decoding. To keep native inference and replace
only decoding, set `Yzma.VideoDecoder`. It must enforce the same track limits
before decoding. The built-in operator checks its returned frame count and
dimensions before passing samples to llama.cpp.

Run the [video example](../../../examples/yzma-video/main.go) from the
repository root. This creates a two-second clip with red then blue and an
ignored AAC soundtrack. The sample command and native test use an FFmpeg
build with the `libx264` encoder.

```sh
ffmpeg -v error \
  -f lavfi -i 'color=red:s=256x256:r=1:d=1[a];color=blue:s=256x256:r=1:d=1[b];[a][b]concat=n=2:v=1:a=0' \
  -f lavfi -i 'sine=frequency=440:duration=2' \
  -map 0:v -map 1:a -c:v libx264 -threads 1 -c:a aac -movflags +faststart colors.mp4

export YZMA_LIB="$HOME/.local/lib/yzma"
task dev:yzma:video -- \
  -model "$HOME/models/SmolVLM-256M-Instruct-Q8_0.gguf" \
  -projector "$HOME/models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf" \
  -video colors.mp4 -question "What color is the first frame?"
```

The `Answer:` line should identify red as the first color. The CPU-only
native test creates clips with both color orders and checks the answers
through Genkit. It does not download models or other media.

```sh
YZMA_TEST_MODEL="$HOME/models/SmolVLM-256M-Instruct-Q8_0.gguf" \
YZMA_TEST_PROJECTOR="$HOME/models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf" \
  task test:yzma:native -- -run '^TestNativeVideoGeneration$' -count=1
```

## Transcribe an inline WAV

Register Qwen3-ASR with `DefineModel`, using `ModelTypeChat` and
`Projector: &yzma.ProjectorOptions{GGUFPath: path, Audio: true}`. Audio pairs
are explicit, not discovered. This path checks for a matching `qwen3a` audio
encoder and verifies native audio support when loading the pair. It does not
require vision, video, or speech output.

Send an `ai.NewMediaPart("audio/wav", "data:audio/wav;base64,...")` in a user
message. `audio/x-wav` is also accepted if the data URL uses that same type.
The WAV must be RIFF with mono, 16 kHz, signed 16-bit PCM. There is no resampling
or compressed-audio decoding. Each clip can be at most 1 MiB before base64 and
30 seconds long. The shared request limits are four media parts and 8 MiB
before base64. Text and audio keep their order through the chat template, and
the resulting tokens must fit the model context. HTTP URLs and file paths are
rejected. Requests can stream text, but cannot use tools or structured output.

Put these public files in `$HOME/models`:

- [Qwen3-ASR-0.6B.Q4_K_S.gguf](https://huggingface.co/mradermacher/Qwen3-ASR-0.6B-GGUF/blob/main/Qwen3-ASR-0.6B.Q4_K_S.gguf)
  (470,781,408 bytes)
- [mmproj-Qwen3-ASR-0.6B-Q8_0.gguf](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/blob/main/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf)
  (214,392,480 bytes)

Use the Yzma-compatible llama.cpp and `mtmd` libraries (tested on CPU with
llama.cpp b12359). From the repository root, generate the sample with FFmpeg
built with the `flite` filter, then run the example:

```sh
export YZMA_LIB="$HOME/.local/lib"
export GOMAXPROCS=2 GOFLAGS=-p=2
export YZMA_TEST_AUDIO_MODEL="$HOME/models/Qwen3-ASR-0.6B.Q4_K_S.gguf"
export YZMA_TEST_AUDIO_PROJECTOR="$HOME/models/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
export YZMA_TEST_AUDIO_WAV="$HOME/models/fox.wav"
ffmpeg -f lavfi -i "flite=text='The quick brown fox jumps over the lazy dog':voice=slt" \
  -ar 16000 -ac 1 -c:a pcm_s16le "$YZMA_TEST_AUDIO_WAV"
task dev:yzma:audio -- -model "$YZMA_TEST_AUDIO_MODEL" \
  -projector "$YZMA_TEST_AUDIO_PROJECTOR" -audio "$YZMA_TEST_AUDIO_WAV"
```

The tested run printed `Transcript: The quick brown fox jumps over the lazy dog.`
The example sets `ChatTemplate` to include Qwen3-ASR's English assistant prefix.
Use a template that accepts string message content so the plugin can insert
each clip at its original position.
Run `task test:yzma:audio:native` with the same exports to check that the
transcript contains words from the sample without matching the exact sentence.
