# Genkit Yzma plugin

Register local GGUF models with Genkit using
[Yzma](https://github.com/hybridgroup/yzma)'s in-process llama.cpp bindings.
No Ollama or vLLM server is required.

Supports metadata-only discovery, chat, raw text completion, and streaming for
decoder-only models. Weights load on demand, and each request gets a fresh
inference context and sampler. Tool calling requires an explicit definition
for a tool-capable chat model with a tool-aware template.
Chat and completion also support schema-constrained JSON objects without tools
or streaming. Media processing is not supported.

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
file with only named templates needs an explicit definition. BERT embedding
models follow the separate [embedding rules](#local-text-embeddings).

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

### Chat versus completion

`ModelDefinition.Type` is required for manual definitions. Choose the
formatting that matches the model.

| Type | Input and formatting | Genkit capabilities |
| --- | --- | --- |
| `yzma.ModelTypeChat` | System, user, and assistant messages formatted with a chat template | Multi-turn and system role |
| `yzma.ModelTypeCompletion` | Exactly one user text message, tokenized directly | Single-turn text completion |

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
| `ModelDefinition.Type` | Required chat or completion mode |
| `ModelDefinition.ToolCalling` | Opt into tool calling for a capable chat model |
| `ModelDefinition.ChatTemplateName` | Select a named template embedded in the GGUF |
| `ModelDefinition.Loading` | GGUF path, GPU placement, and weight-loading modes |
| `ModelDefinition.Context` | Token capacity, batch sizes, and CPU threads |
| `ModelDefinition.GenerationDefaults` | Sampling and output defaults |
| `GenerateContentConfig` | Per-request sampling and output overrides |

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

Generation resolution applies provider defaults, model defaults, then request
overrides to fresh Yzma sampler settings. Output defaults to 1024 tokens. See
[`ExampleModelDefinition`](config_test.go) for configuration and
[`Example_resolveGenerationConfig`](resolve_test.go) for a greedy request
overriding the model's temperature without changing its output limit.

Set `Yzma.Operator` to a custom `ModelOperator` to replace native initialization,
loading, and cleanup for testing or alternate execution environments. To support
chat, that operator must also implement `TextGenerationOperator`.

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
`auto`, `required`, and `none` tool choices are supported.

The adapter uses `message.ParseToolCalls` for Yzma's supported output grammars and
`message.StripMarkup` to preserve accompanying text without tool syntax. Calls
receive unique IDs, which are retained with typed arguments and tool results in
subsequent template renders.

Tool-enabled turns are buffered until parsing completes. Streaming callbacks
receive complete text/tool parts, not raw tool-call syntax. Ordinary chat without
`ToolCalling` continues streaming text incrementally.
Tool inputs must use JSON object schemas. Local schema references are expanded
for rendering; recursive schemas are rejected rather than silently omitted.

Before returning executable calls, the adapter checks offered tool names,
structural completeness, and input schemas. Truncated calls, unknown tools,
invalid arguments, and ambiguous argument types return errors. It does not
execute Yzma's repaired/truncated JSON calls.

Yzma's parser exposes arguments as strings, so their types are restored from the
offered schema rather than guessed. String values such as `"0123"` stay strings.
Unions without a single unambiguous type are not supported. Parsed integers at or
beyond the float64 exact-integer boundary are rejected. Some Gemma parser forms
flatten nested objects; those fail schema validation rather than being executed
with changed argument structure. Use a parser/model format that preserves the
required structure for tools with nested arguments.

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
