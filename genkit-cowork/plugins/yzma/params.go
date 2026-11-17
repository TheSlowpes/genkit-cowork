// Copyright 2026 Kevin Lopes
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package yzma

import (
	"fmt"

	"github.com/hybridgroup/yzma/pkg/llama"
)

// SplitMode controls placement of model weights across GPUs.
type SplitMode string

const (
	SplitModeNone   SplitMode = "none"  // single GPU
	SplitModeLayer  SplitMode = "layer" // split layers and KV across GPUs
	SplitModeRow    SplitMode = "row"   // split layers and KV across GPUs, use tensor parallelism if supported
	SplitModeTensor SplitMode = "tensor"
)

// resolveSplitMode preserves the supplied native default for an omitted mode.
func resolveSplitMode(mode SplitMode, fallback llama.SplitMode) (llama.SplitMode, error) {
	switch mode {
	case "":
		return fallback, nil
	case SplitModeNone:
		return llama.SplitModeNone, nil
	case SplitModeLayer:
		return llama.SplitModeLayer, nil
	case SplitModeRow:
		return llama.SplitModeRow, nil
	case SplitModeTensor:
		return llama.SplitModeTensor, nil
	default:
		return 0, fmt.Errorf("invalid split mode: %s", mode)
	}
}

// LoadMode controls how llama.cpp reads model weights.
type LoadMode string

const (
	LoadModeAuto      LoadMode = "auto"       // auto-detect based on device capabilities
	LoadModeNone      LoadMode = "none"       // no special loading mode
	LoadModeMmap      LoadMode = "mmap"       // memory map the model
	LoadModeMlock     LoadMode = "mlock"      // force system to keep model in RAM rather than swapping or compressing
	LoadModeMmapmLock LoadMode = "mmap-mlock" // memory map the model and force system to keep it in RAM
	LoadModeDirectIO  LoadMode = "direct-io"  // use direct I/O if available
)

// resolveLoadMode preserves the supplied native default for an omitted mode.
func resolveLoadMode(mode LoadMode, fallback llama.LoadMode) (llama.LoadMode, error) {
	switch mode {
	case "":
		return fallback, nil
	case LoadModeAuto:
		return llama.LoadModeAuto, nil
	case LoadModeNone:
		return llama.LoadModeNone, nil
	case LoadModeMmap:
		return llama.LoadModeMmap, nil
	case LoadModeMlock:
		return llama.LoadModeMlock, nil
	case LoadModeMmapmLock:
		return llama.LoadModeMmapMlock, nil
	case LoadModeDirectIO:
		return llama.LoadModeDirectIO, nil
	default:
		return 0, fmt.Errorf("invalid load mode: %s", mode)
	}
}

// LazyMode controls on-demand reads of tensors, not plugin initialization.
type LazyMode string

const (
	LazyModeOff  LazyMode = "off"  // always read the whole tensor up front
	LazyModeAuto LazyMode = "auto" // lazy only for marked tensors larger than 4 GiB (requires mmap)
	LazyModeOn   LazyMode = "on"   // read the rows of tensors marked by the arch on demand (requires mmap)
)

// resolveLazyMode preserves the supplied native default for an omitted mode.
func resolveLazyMode(mode LazyMode, fallback llama.LazyMode) (llama.LazyMode, error) {
	switch mode {
	case "":
		return fallback, nil
	case LazyModeOff:
		return llama.LazyModeOff, nil
	case LazyModeAuto:
		return llama.LazyModeAuto, nil
	case LazyModeOn:
		return llama.LazyModeOn, nil
	default:
		return 0, fmt.Errorf("invalid lazy mode: %s", mode)
	}
}
