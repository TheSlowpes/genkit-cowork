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
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/ardanlabs/jinja"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/hybridgroup/yzma/pkg/download"
)

// ListActions describes the discovered models without resolving them or loading
// weights. Genkit lists explicit definitions separately.
func (y *Yzma) ListActions(ctx context.Context) []api.ActionDesc {
	y.mu.Lock()
	defer y.mu.Unlock()
	if y.closed {
		return nil
	}
	var actions []api.ActionDesc
	for _, name := range slices.Sorted(maps.Keys(y.catalog)) {
		if ctx.Err() != nil {
			return nil
		}
		if gen := y.models[name]; gen == nil || !gen.explicit {
			model := y.catalog[name]
			// Derive the descriptor using the same schema and capability metadata
			// as resolution, without installing a generator or action.
			actions = append(actions, y.newDiscoveredAction(model, nil).Desc())
		}
	}
	return actions
}

// ResolveAction constructs a model or embedder for Genkit to register. The plugin retains
// the generator that owns the weights until Close, even across registry scopes.
func (y *Yzma) ResolveAction(atype api.ActionType, name string) api.Action {
	y.mu.Lock()
	defer y.mu.Unlock()
	model, ok := y.catalog[name]
	if !ok || y.closed || atype != model.actionType() {
		return nil
	}
	gen := y.models[name]
	if gen != nil && gen.explicit {
		return nil
	}
	if gen == nil {
		gen = y.newGenerator(model)
		y.models[name] = gen
	}
	return y.newDiscoveredAction(model, gen)
}

func (y *Yzma) newDiscoveredAction(model ModelDefinition, gen *generator) api.Action {
	if model.actionType() == api.ActionTypeEmbedder {
		var embed ai.EmbedderFunc
		if gen != nil {
			embed = gen.embed
		}
		return ai.NewEmbedder(api.NewName(provider, model.Name), model.embedding.options(model.Name), embed).(api.Action)
	}
	var generate ai.ModelFunc
	if gen != nil {
		generate = gen.generate
	}
	// Genkit's ai.Model interface only exposes Generate and Register, but
	// ai.NewModel returns a concrete action, as used by Genkit's own plugins.
	return ai.NewModel(api.NewName(provider, model.Name), y.modelMetadata(model, nil), generate).(api.Action)
}

var (
	modelIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)
	shardPattern   = regexp.MustCompile(`^(.+)-([0-9]{5})-of-([0-9]{5})\.gguf$`)
)

type ggufCandidate struct {
	path   string
	header ggufHeader
}

func discoverModels(ctx context.Context, dir string) map[string]ModelDefinition {
	useDefault := dir == ""
	if useDefault {
		dir = download.DefaultModelsDir()
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		slog.WarnContext(ctx, "yzma: cannot resolve models directory", "path", dir, "error", err)
		return nil
	}
	entries, err := os.ReadDir(abs)
	if err != nil {
		// Only silence a genuinely absent default, not a dangling symlink.
		if _, statErr := os.Lstat(abs); !useDefault || !errors.Is(statErr, fs.ErrNotExist) || !errors.Is(err, fs.ErrNotExist) {
			slog.WarnContext(ctx, "yzma: cannot read models directory", "path", abs, "error", err)
		}
		return nil
	}
	skip := func(path string, reason error) {
		slog.WarnContext(ctx, "yzma: skipping GGUF", "path", path, "reason", reason)
	}
	namesByID := make(map[string][]string)
	files := make(map[string]ggufCandidate)
	for _, entry := range entries {
		if ctx.Err() != nil {
			return nil
		}
		filename := entry.Name()
		ext := filepath.Ext(filename)
		if !strings.EqualFold(ext, ".gguf") {
			continue
		}
		stem := strings.TrimSuffix(filename, ext)
		path := filepath.Join(abs, filename)
		if !modelIDPattern.MatchString(stem) {
			skip(path, errors.New("filename must form an ASCII model ID using letters, digits, dots, underscores or hyphens"))
			continue
		}
		id := strings.ToLower(stem)
		namesByID[id] = append(namesByID[id], filename)
	}
	for _, id := range slices.Sorted(maps.Keys(namesByID)) {
		names := namesByID[id]
		if len(names) > 1 {
			for _, filename := range names {
				skip(filepath.Join(abs, filename), fmt.Errorf("ambiguous model ID %q", id))
			}
			continue
		}
		filename := names[0]
		path := filepath.Join(abs, filename)
		h, err := readGGUFHeader(ctx, path)
		if err != nil {
			skip(path, err)
			continue
		}
		files[filename] = ggufCandidate{path, h}
	}
	catalog := make(map[string]ModelDefinition)
	for _, filename := range slices.Sorted(maps.Keys(files)) {
		file := files[filename]
		tensors, err := modelTensors(filename, files)
		if err != nil {
			skip(file.path, err)
			continue
		}
		id := strings.TrimSuffix(strings.ToLower(filename), ".gguf")
		model := ModelDefinition{Name: id, Loading: ModelOptions{GGUFPath: file.path}}
		if spec, ok := discoverEmbedding(file.header.values, tensors); ok {
			model.embedding = spec
			catalog[id] = model
			continue
		}
		typ, err := classifyTextModel(file.header.values, tensors)
		if err != nil {
			skip(file.path, err)
			continue
		}
		model.Type = typ
		catalog[id] = model
	}
	return catalog
}

func modelTensors(filename string, files map[string]ggufCandidate) (map[string]bool, error) {
	h := files[filename].header
	no, hasNo := h.values["split.no"]
	count, hasCount := h.values["split.count"]
	shard := shardPattern.FindStringSubmatch(filename)
	// llama-gguf-split can leave split.count=0 in a merged file.
	notSplit := (!hasNo || no == (ggufValue{kind: 2})) && (!hasCount || (count.kind == 2 && count.num <= 1))
	if notSplit && shard == nil && !shardPattern.MatchString(strings.ToLower(filename)) {
		return h.tensors, nil
	}
	if no.num > 0 || (shard != nil && shard[2] != "00001") {
		return nil, errors.New("secondary GGUF shard")
	}
	if shard == nil || !hasNo || !hasCount || no.kind != 2 || count.kind != 2 ||
		count.num < 2 || count.num > 256 || shard[3] != fmt.Sprintf("%05d", count.num) {
		return nil, errors.New("ambiguous GGUF split metadata or filename")
	}
	total := h.values["split.tensors.count"]
	if total.kind != 5 || total.num == 0 {
		return nil, errors.New("missing or invalid GGUF split tensor count")
	}
	tensors := maps.Clone(h.tensors)
	for i := uint64(1); i < count.num; i++ {
		name := fmt.Sprintf("%s-%05d-of-%s.gguf", shard[1], i+1, shard[3])
		part, ok := files[name]
		other := part.header.values
		arch, hasArch := other["general.architecture"]
		otherTotal, hasTotal := other["split.tensors.count"]
		// llama-gguf-split writes model metadata only to the first file.
		if !ok || other["split.no"] != (ggufValue{kind: 2, num: i}) ||
			other["split.count"] != count || (hasTotal && otherTotal != total) ||
			(hasArch && arch != h.values["general.architecture"]) {
			return nil, fmt.Errorf("missing or incompatible GGUF shard %q", name)
		}
		for tensor := range part.header.tensors {
			if tensors[tensor] {
				return nil, fmt.Errorf("duplicate GGUF tensor %q in shard %q", tensor, name)
			}
			tensors[tensor] = true
		}
	}
	if uint64(len(tensors)) != total.num {
		return nil, fmt.Errorf("GGUF split contains %d tensors, expected %d", len(tensors), total.num)
	}
	return tensors, nil
}

func classifyTextModel(values map[string]ggufValue, tensors map[string]bool) (ModelType, error) {
	typ := values["general.type"].text
	arch := values["general.architecture"].text
	_, hasProjector := values["clip.projector_type"]
	if typ == "mmproj" || arch == "clip" || hasProjector {
		return "", errors.New("multimodal projector")
	}
	unclassified := errors.New("unclassified GGUF; use DefineModel or DefineEmbedder to choose its role")
	// Start with decoder architectures exercised by the text path. A template
	// alone cannot distinguish a decoder from an embedder, projector or adapter.
	if typ != "model" || (arch != "llama" && arch != "gemma4") {
		return "", unclassified
	}
	causal, hasCausal := values[arch+".attention.causal"]
	_, hasPooling := values[arch+".pooling_type"]
	if hasPooling || (hasCausal && (causal.kind != 7 || causal.num != 1)) ||
		tensors["cls.weight"] || tensors["cls.output.weight"] {
		return "", unclassified
	}
	tokenizer := values["tokenizer.ggml.model"].text
	tokens := values["tokenizer.ggml.tokens"]
	if tokenizer == "" || tokenizer == "none" || tokens.kind != ggufArray || tokens.element != ggufString || tokens.num == 0 ||
		!tensors["token_embd.weight"] || !tensors["blk.0.attn_q.weight"] || !tensors["output_norm.weight"] {
		return "", unclassified
	}
	if template, ok := values["tokenizer.chat_template"]; ok {
		if template.kind != ggufString || strings.TrimSpace(template.text) == "" || strings.ContainsRune(template.text, '\x00') {
			return "", errors.New("invalid default chat template")
		}
		if _, err := jinja.Compile(template.text); err != nil {
			return "", fmt.Errorf("invalid default chat template: %w", err)
		}
		return ModelTypeChat, nil
	}
	for key := range values {
		if strings.HasPrefix(key, "tokenizer.chat_template.") {
			return "", errors.New("only named chat templates are present; select one with DefineModel")
		}
	}
	return ModelTypeCompletion, nil
}
