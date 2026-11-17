# Genkit Yzma plugin

Register local GGUF models with Genkit using
[Yzma](https://github.com/hybridgroup/yzma)'s in-process llama.cpp bindings.
No Ollama or vLLM server is required.

Configuration, lazy weight loading, explicit preloading, and cleanup work.
**Token generation is not implemented yet.** A generation call loads the weights,
then returns `ErrGenerationNotImplemented`. Inference-context allocation,
streaming, tool calling, and media processing are not implemented.

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
		Type: "chat",
		Loading: yzma.ModelOptions{
			GGUFPath:   "/path/to/model.gguf",
			NGPULayers: &cpuLayers,
		},
	}, nil)

	if err := local.Preload(ctx, "local-chat"); err != nil {
		log.Print(err)
		return
	}
	fmt.Println("Model weights loaded.")
}
```

`DefineModel` validates and snapshots configuration without loading weights.
`Preload` takes the unprefixed registration name. The Genkit model identifier is
`yzma/local-chat`; lookup helpers are `yzma.Model(g, "local-chat")` and
`yzma.IsDefinedModel(g, "local-chat")`.

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
| `ModelDefinition.Loading` | GGUF path, GPU placement, and weight-loading modes |
| `ModelDefinition.Context` | Token capacity, batch sizes, and CPU threads |
| `ModelDefinition.GenerationDefaults` | Sampling and output defaults |
| `GenerateContentConfig` | Per-request sampling and output overrides |

Loading options are used now. Context and generation settings have resolvers but
are not connected to inference yet. `ModelsDir` and `Verbose` are reserved;
automatic discovery and logging configuration are not implemented.

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
loading, and cleanup for testing or alternate execution environments.

## Run the example

From the repository root, load and release a model on the CPU without inference:

```sh
YZMA_LIB=/path/to/libs go run ./examples/yzma-loading -model /path/to/model.gguf
```

The [example](../../../examples/yzma-loading/main.go) handles cancellation and
cleans up with a separate timeout.

## Tests

Run from the repository root. Unit tests use a custom `ModelOperator` and require
no native installation:

```sh
go test -race ./genkit-cowork/plugins/yzma
```

To exercise actual loading with existing native libraries and a local model:

```sh
YZMA_LIB=/path/to/libs YZMA_TEST_MODEL=/path/to/model.gguf \
  go test ./genkit-cowork/plugins/yzma -run TestNativeModelLoading -v
```

The native test skips unless both environment variables are set. It does not
download models or run token generation.
