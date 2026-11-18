# Genkit Yzma plugin

Register local GGUF models with Genkit using
[Yzma](https://github.com/hybridgroup/yzma)'s in-process llama.cpp bindings.
No Ollama or vLLM server is required.

Supports lazy weight loading, explicit preloading, chat, raw text completion,
streaming, and cleanup for decoder-only models. Each request gets a fresh inference context and
sampler. Tool calling is available for chat models with a tool-aware template.
Media processing and constrained structured output are not supported yet;
unsupported requests return errors.

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

Import only the plugin and Genkit. No other cowork components need registration:

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
`Preload` takes the unprefixed registration name. The Genkit model identifier is
`yzma/local-chat`; lookup helpers are `yzma.Model(g, "local-chat")` and
`yzma.IsDefinedModel(g, "local-chat")`.
Preloading is optional: the first generation call loads the model when necessary.

### Chat versus completion

`ModelDefinition.Type` is required. Choose explicitly; the plugin does not infer
the mode from a model filename or the presence of template metadata.

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
`Preload` or generation.

Call `Close(ctx)` to reject new work, wait for active model use, and free weights.
If cleanup times out or fails, call `Close` again. Shared native libraries and
backends stay initialized for the process lifetime. All default-operator plugin
instances must use the same library directory. Do not independently unload
Yzma's native runtime while the plugin is in use.

## Configuration

| Configuration | Scope |
| --- | --- |
| `Yzma.LibPath` | Native library directory; falls back to `YZMA_LIB` |
| `ModelDefinition.Type` | Required chat or completion mode |
| `ModelDefinition.ToolCalling` | Opt into tool calling for a capable chat model |
| `ModelDefinition.ChatTemplateName` | Select a named template embedded in the GGUF |
| `ModelDefinition.Loading` | GGUF path, GPU placement, and weight-loading modes |
| `ModelDefinition.Context` | Token capacity, batch sizes, and CPU threads |
| `ModelDefinition.GenerationDefaults` | Sampling and output defaults |
| `GenerateContentConfig` | Per-request sampling and output overrides |

Context and generation settings are applied on each request. `ModelsDir` and
`Verbose` are reserved; automatic discovery and logging configuration are not
implemented.

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

## Chat in the Genkit Dev UI

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
without an embedded chat template.

To test the embedded template without allocating model weights:

```sh
YZMA_LIB=/path/to/libs YZMA_TEST_MODEL=~/models/gemma-4-E4B-it-Q4_0.gguf \
  task test:yzma:native -- -run '^TestNativeEmbeddedTemplate$'
```
