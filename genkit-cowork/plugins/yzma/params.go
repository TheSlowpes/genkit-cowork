package yzma

import "github.com/hybridgroup/yzma/pkg/llama"

type SplitMode string

const (
	SplitModeNone   SplitMode = "none"  // single GPU
	SplitModeLayer  SplitMode = "layer" // split layers and KV across GPUs
	SplitModeRow    SplitMode = "row"   // split layers and KV across GPUs, use tensor parallelism if supported
	SplitModeTensor SplitMode = "tensor"
)

func resolveSplitMode(mode SplitMode) llama.SplitMode {
	switch mode {
	case SplitModeLayer:
		return llama.SplitModeLayer
	case SplitModeRow:
		return llama.SplitModeRow
	case SplitModeTensor:
		return llama.SplitModeTensor
	default:
		return llama.SplitModeNone
	}
}

type LoadMode string

const (
	LoadModeAuto      LoadMode = "auto"       // auto-detect based on device capabilities
	LoadModeNone      LoadMode = "none"       // no special loading mode
	LoadModeMmap      LoadMode = "mmap"       // memory map the model
	LoadModeMlock     LoadMode = "mlock"      // force system to keep model in RAM rather than swapping or compressing
	LoadModeMmapmLock LoadMode = "mmap-mlock" // memory map the model and force system to keep it in RAM
	LoadModeDirectIO  LoadMode = "direct-io"  // use direct I/O if available
)

func resolveLoadMode(mode LoadMode) llama.LoadMode {
	switch mode {
	case LoadModeAuto:
		return llama.LoadModeAuto
	case LoadModeMmap:
		return llama.LoadModeMmap
	case LoadModeMlock:
		return llama.LoadModeMlock
	case LoadModeMmapmLock:
		return llama.LoadModeMmapMlock
	case LoadModeDirectIO:
		return llama.LoadModeDirectIO
	default:
		return llama.LoadModeNone
	}
}

type LazyMode string

const (
	LazyModeOff  LazyMode = "off"  // always read the whole tensor up front
	LazyModeAuto LazyMode = "auto" // lazy only for marked tensors larger than 4 GiB (requires mmap)
	LazyModeOn   LazyMode = "on"   // read the rows of tensors marked by the arch on demand (requires mmap)
)

func resolveLazyMode(mode LazyMode) llama.LazyMode {
	switch mode {
	case LazyModeAuto:
		return llama.LazyModeAuto
	case LazyModeOn:
		return llama.LazyModeOn
	default:
		return llama.LazyModeOff
	}
}
