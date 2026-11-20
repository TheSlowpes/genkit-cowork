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
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"testing/fstest"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/go-cmp/cmp"
	"github.com/hybridgroup/yzma/pkg/llama"
)

type memoryCatalog struct{ files fstest.MapFS }

func (c memoryCatalog) ReadDir(ctx context.Context, _ string) ([]fs.DirEntry, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return fs.ReadDir(c.files, ".")
}

func (c memoryCatalog) Lstat(ctx context.Context, path string) (fs.FileInfo, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return fs.Stat(c.files, filepath.Base(path))
}

func (c memoryCatalog) OpenRegular(ctx context.Context, path string) (io.ReadCloser, int64, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	file, err := c.files.Open(filepath.Base(path))
	if err != nil {
		return nil, 0, err
	}
	info, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, 0, err
	}
	return file, info.Size(), nil
}

func TestCatalogOperatorCanSupplyVirtualGGUFs(t *testing.T) {
	path := writeGGUFFixture(t, t.TempDir(), "virtual.gguf", nil)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	local := &Yzma{
		LibPath: t.TempDir(), ModelsDir: filepath.Join(t.TempDir(), "absent"),
		CatalogOperator: memoryCatalog{fstest.MapFS{
			"virtual.gguf":       &fstest.MapFile{Data: data},
			"broken.gguf":        &fstest.MapFile{Data: data[:8]},
			"judge.gguf":         &fstest.MapFile{Data: data},
			"judge.decider.json": &fstest.MapFile{Data: []byte(`{"temperature":1.03}`)},
		}},
		Operator: &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) {
			t.Fatal("catalog access must not load model weights")
			return 0, nil
		}},
	}
	g := genkit.Init(t.Context(), genkit.WithPlugins(local))
	t.Cleanup(func() {
		if err := local.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	if Model(g, "virtual") == nil || Model(g, "broken") != nil {
		t.Fatal("discovery should expose only the valid virtual GGUF")
	}
	if got := local.catalog["judge"]; got.Type != ModelTypeDecision || got.Decision.Readout != DecisionReadoutDecider {
		t.Fatalf("virtual decision discovery = %+v, want a decider", got)
	}
}

func writeGGUFFixture(t *testing.T, dir, filename string, metadata map[string]ggufValue) string {
	t.Helper()
	values := map[string]ggufValue{
		"general.type":          {kind: ggufString, text: "model"},
		"general.architecture":  {kind: ggufString, text: "llama"},
		"tokenizer.ggml.model":  {kind: ggufString, text: "gpt2"},
		"tokenizer.ggml.tokens": {kind: ggufArray},
	}
	for k, v := range metadata {
		values[k] = v
	}
	return writeGGUF(t, dir, filename, values, "token_embd.weight", "blk.0.attn_q.weight", "output_norm.weight")
}

func writeGGUF(t *testing.T, dir, filename string, values map[string]ggufValue, tensors ...string) string {
	t.Helper()
	var out bytes.Buffer
	write := func(v any) {
		t.Helper()
		if err := binary.Write(&out, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	text := func(s string) {
		write(uint64(len(s)))
		out.WriteString(s)
	}
	out.WriteString("GGUF")
	write(uint32(3))
	write(uint64(len(tensors)))
	write(uint64(len(values)))
	for k, v := range values {
		text(k)
		write(v.kind)
		switch v.kind {
		case ggufString:
			text(v.text)
		case ggufArray:
			write(uint32(ggufString))
			write(uint64(2))
			text("hello")
			text(" world")
		case 2:
			write(uint16(v.num))
		case 4:
			write(uint32(v.num))
		case 5:
			write(int32(v.num))
		case 7:
			write(uint8(v.num))
		default:
			t.Fatalf("invalid fixture metadata: %+v", v)
		}
	}
	for i, name := range tensors {
		text(name)
		write(uint32(1))      // Number of dimensions.
		write(uint64(1))      // Elements in the dimension.
		write(uint32(0))      // GGML F32 type.
		write(uint64(i * 32)) // Aligned offset into tensor data.
	}
	out.Write(make([]byte, (32-out.Len()%32)%32+len(tensors)*32))
	path := filepath.Join(dir, filename)
	if err := os.WriteFile(path, out.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func initDiscoveryTest(t *testing.T, dir string, op ModelOperator) (*Yzma, *genkit.Genkit) {
	t.Helper()
	plugin := &Yzma{LibPath: t.TempDir(), ModelsDir: dir, Operator: op}
	g := genkit.Init(t.Context(), genkit.WithPlugins(plugin))
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.WithoutCancel(t.Context()), 5*time.Second)
		defer cancel()
		if err := plugin.Close(ctx); err != nil {
			t.Errorf("close plugin: %v", err)
		}
	})
	return plugin, g
}

func discoveryNames(ctx context.Context, p *Yzma) []string {
	var names []string
	for _, d := range p.ListActions(ctx) {
		names = append(names, d.Name)
	}
	return names
}

func TestDiscoverTextCatalog(t *testing.T) {
	dir := t.TempDir()
	var logs bytes.Buffer
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, nil)))
	t.Cleanup(func() { slog.SetDefault(old) })
	chat := ggufValue{kind: ggufString, text: "{% for message in messages %}{{ message.content }}{% endfor %}"}
	writeGGUFFixture(t, dir, "Example.Q4.gguf", map[string]ggufValue{"tokenizer.chat_template": chat})
	writeGGUFFixture(t, dir, "Example.Q8.GGUF", nil)
	writeGGUFFixture(t, dir, "attachment.gguf", map[string]ggufValue{
		"general.architecture": {kind: ggufString, text: "clip"}, "tokenizer.chat_template": chat,
	})
	writeGGUFFixture(t, dir, "unknown.gguf", map[string]ggufValue{
		"general.architecture": {kind: ggufString, text: "unknown"}, "tokenizer.chat_template": chat,
	})
	writeGGUFFixture(t, dir, "embedder.gguf", map[string]ggufValue{
		"llama.pooling_type": {kind: 4, num: 1}, "tokenizer.chat_template": chat,
	})
	writeGGUFFixture(t, dir, "named-only.gguf", map[string]ggufValue{"tokenizer.chat_template.tool_use": chat})
	writeGGUFFixture(t, dir, "bad-template.gguf", map[string]ggufValue{
		"tokenizer.chat_template": {kind: ggufString, text: "{% bad_syntax %}"},
	})
	a := writeGGUFFixture(t, dir, "collision.gguf", nil)
	b := writeGGUFFixture(t, dir, "collision.GGUF", nil)
	fileA, err := os.Stat(a)
	if err != nil {
		t.Fatal(err)
	}
	fileB, err := os.Stat(b)
	if err != nil {
		t.Fatal(err)
	}
	caseSensitive := !os.SameFile(fileA, fileB)
	if !caseSensitive {
		if err := os.Remove(a); err != nil {
			t.Fatal(err)
		}
	}
	writeGGUFFixture(t, dir, "split-00001-of-00002.gguf", map[string]ggufValue{
		"split.no": {kind: 2}, "split.count": {kind: 2, num: 2}, "split.tensors.count": {kind: 5, num: 4},
	})
	writeGGUF(t, dir, "split-00002-of-00002.gguf", map[string]ggufValue{
		"split.no": {kind: 2, num: 1}, "split.count": {kind: 2, num: 2}, "split.tensors.count": {kind: 5, num: 4},
	}, "blk.1.attn_q.weight")
	writeGGUFFixture(t, dir, "incomplete-00001-of-00002.gguf", map[string]ggufValue{
		"split.no": {kind: 2}, "split.count": {kind: 2, num: 2}, "split.tensors.count": {kind: 5, num: 4},
	})
	truncated := writeGGUFFixture(t, dir, "truncated.gguf", nil)
	if err := os.Truncate(truncated, 24); err != nil {
		t.Fatal(err)
	}

	plugin, g := initDiscoveryTest(t, dir, &testModelOperator{
		load: func(context.Context, ModelOptions) (llama.Model, error) {
			t.Fatal("listing and resolution must not load weights")
			return 0, nil
		},
	})
	want := []string{"yzma/example.q4", "yzma/example.q8", "yzma/split-00001-of-00002"}
	if diff := cmp.Diff(want, discoveryNames(t.Context(), plugin)); diff != "" {
		t.Errorf("listed names (-want +got):\n%s", diff)
	}
	for _, d := range plugin.ListActions(t.Context()) {
		if d.Type != api.ActionTypeModel || d.Key != "/model/"+d.Name || d.InputSchema == nil {
			t.Errorf("invalid descriptor: %+v", d)
		}
		info := d.Metadata["model"].(map[string]any)
		supports := info["supports"].(map[string]any)
		if supports["tools"] != false || supports["multiturn"] != (d.Name == "yzma/example.q4") ||
			supports["constrained"] != ai.ConstrainedSupport("") {
			t.Errorf("%q supports = %v", d.Name, supports)
		}
	}
	diagnostics := []string{
		"multimodal projector", "unclassified GGUF", "secondary GGUF shard",
		"missing or incompatible GGUF shard", "invalid default chat template",
		"only named chat templates", "read GGUF metadata key",
	}
	if caseSensitive {
		diagnostics = append(diagnostics, "ambiguous model ID")
	}
	for _, diagnostic := range diagnostics {
		if !strings.Contains(logs.String(), diagnostic) {
			t.Errorf("diagnostics missing %q:\n%s", diagnostic, logs.String())
		}
	}
	if model := Model(g, "unknown"); model != nil {
		t.Fatal("unclassified model resolved without a manual definition")
	}
	manual := testModelDefinition()
	manual.Name, manual.Loading.GGUFPath = "unknown", filepath.Join(dir, "unknown.gguf")
	plugin.DefineModel(g, manual, nil)
	if Model(g, manual.Name) == nil {
		t.Fatal("manual registration of unclassified model did not resolve")
	}
	writeGGUFFixture(t, dir, "added.gguf", nil)
	if Model(g, "added") != nil {
		t.Error("catalog changed after initialization")
	}
}

func TestDiscoveredModelSupportsConstrainedOutput(t *testing.T) {
	dir := t.TempDir()
	writeGGUFFixture(t, dir, "discovered.gguf", nil)
	type result struct {
		Code string `json:"code" jsonschema:"enum=YZMA"`
	}
	op := &constrainedTestOperator{
		testTextOperator: &testTextOperator{testModelOperator: &testModelOperator{
			load: func(context.Context, ModelOptions) (llama.Model, error) { return 42, nil },
		}},
		generateConstrained: func(input TextGenerationInput, _ string) TextGenerationResult {
			if input.Type != ModelTypeCompletion || input.ToolCalling {
				t.Errorf("generation mode = %q, tool calling = %v; want completion without tools", input.Type, input.ToolCalling)
			}
			return TextGenerationResult{Text: `{"code":"YZMA"}`}
		},
	}
	plugin, g := initDiscoveryTest(t, dir, op)
	actions := plugin.ListActions(t.Context())
	if len(actions) != 1 {
		t.Fatalf("listed %d actions, want 1", len(actions))
	}
	supports := actions[0].Metadata["model"].(map[string]any)["supports"].(map[string]any)
	if supports["constrained"] != ai.ConstrainedSupportNoTools || supports["tools"] != false {
		t.Fatalf("discovered supports = %v, want constrained output without tools", supports)
	}
	got, response, err := genkit.GenerateData[result](t.Context(), g,
		ai.WithModelName("yzma/discovered"), ai.WithPrompt("Return the code."))
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.Code != "YZMA" || response.Request.Output == nil || !response.Request.Output.Constrained {
		t.Fatalf("got %+v, want code YZMA from a constrained request", got)
	}
}

func TestDiscoveryDirectorySelection(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	var logs bytes.Buffer
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, nil)))
	t.Cleanup(func() { slog.SetDefault(old) })
	defaultDir := filepath.Join(home, "models")
	op := &testModelOperator{}
	plugin, _ := initDiscoveryTest(t, "", op)
	if len(plugin.ListActions(t.Context())) != 0 || strings.Contains(logs.String(), "cannot read models directory") {
		t.Fatalf("absent default must be quiet and empty, logs:\n%s", logs.String())
	}
	if err := os.Mkdir(defaultDir, 0700); err != nil {
		t.Fatal(err)
	}
	writeGGUFFixture(t, defaultDir, "default.gguf", nil)
	plugin, _ = initDiscoveryTest(t, "", op)
	if diff := cmp.Diff([]string{"yzma/default"}, discoveryNames(t.Context(), plugin)); diff != "" {
		t.Errorf("default catalog (-want +got):\n%s", diff)
	}
	override := t.TempDir()
	file := writeGGUFFixture(t, override, "override.gguf", nil)
	plugin, _ = initDiscoveryTest(t, override, op)
	if diff := cmp.Diff([]string{"yzma/override"}, discoveryNames(t.Context(), plugin)); diff != "" {
		t.Errorf("explicit catalog (-want +got):\n%s", diff)
	}
	for _, bad := range []string{file, filepath.Join(override, "missing")} {
		logs.Reset()
		plugin, _ = initDiscoveryTest(t, bad, op)
		if len(plugin.ListActions(t.Context())) != 0 || !strings.Contains(logs.String(), "cannot read models directory") ||
			!strings.Contains(logs.String(), bad) {
			t.Errorf("invalid directory %q must be diagnosed, logs:\n%s", bad, logs.String())
		}
	}
}

func TestDiscoveryResolvesBeforeListingAndKeepsExplicitOverride(t *testing.T) {
	dir := t.TempDir()
	path := writeGGUFFixture(t, dir, "discovered.gguf", nil)
	writeGGUFFixture(t, dir, "override.gguf", nil)
	var loaded []string
	var freed []llama.Model
	op := &testTextOperator{
		testModelOperator: &testModelOperator{
			load: func(_ context.Context, opts ModelOptions) (llama.Model, error) {
				loaded = append(loaded, opts.GGUFPath)
				return llama.Model(len(loaded)), nil
			},
			free: func(_ context.Context, model llama.Model) error {
				freed = append(freed, model)
				return nil
			},
		},
		generate: func(ctx context.Context, model llama.Model, input TextGenerationInput, onText func(context.Context, string) error) (TextGenerationResult, error) {
			want := ModelTypeCompletion
			if model == 2 {
				want = ModelTypeChat
			}
			if input.Type != want {
				t.Errorf("model %d type = %q, want %q", model, input.Type, want)
			}
			if err := onText(ctx, "world"); err != nil {
				return TextGenerationResult{}, err
			}
			return TextGenerationResult{Text: "world", InputTokens: 1, OutputTokens: 1}, nil
		},
	}
	plugin, g := initDiscoveryTest(t, dir, op)
	manual := testModelDefinition()
	manual.Name = "override"
	plugin.DefineModel(g, manual, &ai.ModelOptions{Label: "My chat override"})
	model := Model(g, "discovered")
	if model == nil || len(loaded) != 0 {
		t.Fatalf("resolve before listing: model=%v loads=%v", model != nil, loaded)
	}
	if diff := cmp.Diff([]string{"yzma/discovered"}, discoveryNames(t.Context(), plugin)); diff != "" {
		t.Errorf("catalog with explicit override (-want +got):\n%s", diff)
	}
	for _, name := range []string{"discovered", "discovered", "override"} {
		var streamed strings.Builder
		model := Model(g, name)
		resp, err := model.Generate(t.Context(), ai.NewModelRequest(nil, ai.NewUserTextMessage("hello")),
			func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				for _, part := range chunk.Content {
					streamed.WriteString(part.Text)
				}
				return nil
			})
		if err != nil {
			t.Fatal(err)
		}
		if resp.Text() != "world" || streamed.String() != resp.Text() {
			t.Errorf("%s: response = %q, streamed = %q", name, resp.Text(), streamed.String())
		}
	}
	if diff := cmp.Diff([]string{path, manual.Loading.GGUFPath}, loaded); diff != "" {
		t.Errorf("loaded paths (-want +got):\n%s", diff)
	}
	for range 2 {
		if err := plugin.Close(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	if diff := cmp.Diff(map[llama.Model]int{1: 1, 2: 1}, countFreed(freed)); diff != "" {
		t.Errorf("freed handles (-want +got):\n%s", diff)
	}
	if _, err := model.Generate(t.Context(), ai.NewModelRequest(nil, ai.NewUserTextMessage("hello")), nil); !errors.Is(err, ErrClosed) {
		t.Errorf("cached discovered model after close: %v, want ErrClosed", err)
	}
}

func countFreed(freed []llama.Model) map[llama.Model]int {
	counts := make(map[llama.Model]int)
	for _, handle := range freed {
		counts[handle]++
	}
	return counts
}

func TestManualAndDiscoveredLoadsShareLimit(t *testing.T) {
	dir := t.TempDir()
	writeGGUFFixture(t, dir, "discovered.gguf", nil)
	entered, allow := make(chan string, 2), make(chan struct{})
	op := &testModelOperator{load: func(ctx context.Context, opts ModelOptions) (llama.Model, error) {
		entered <- filepath.Base(opts.GGUFPath)
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-allow:
			return 1, nil
		}
	}}
	plugin, g := initDiscoveryTest(t, dir, op)
	plugin.DefineModel(g, testModelDefinition(), nil)
	if Model(g, "discovered") == nil {
		t.Fatal("model not resolved")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	first := make(chan error, 1)
	go func() { first <- plugin.Preload(ctx, "local-chat") }()
	select {
	case <-entered:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	wait, stop := context.WithTimeout(ctx, 30*time.Millisecond)
	defer stop()
	err := plugin.Preload(wait, "discovered")
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("waiting load error = %v, want deadline exceeded", err)
	}
	select {
	case path := <-entered:
		t.Errorf("queued load started before the active load finished: %q", path)
	default:
	}
	close(allow)
	if err := <-first; err != nil {
		t.Fatal(err)
	}
	if err := plugin.Preload(ctx, "discovered"); err != nil {
		t.Fatal(err)
	}
	select {
	case path := <-entered:
		if path != "discovered.gguf" {
			t.Errorf("loaded %q, want discovered.gguf", path)
		}
	default:
		t.Error("queued model did not load on retry")
	}
}

func TestNativeDiscoveredTextGeneration(t *testing.T) {
	path, lib := os.Getenv("YZMA_TEST_MODEL"), os.Getenv("YZMA_LIB")
	if path == "" || lib == "" {
		t.Skip("set YZMA_TEST_MODEL and YZMA_LIB to run native discovery")
	}
	path, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	name := strings.TrimSuffix(strings.ToLower(filepath.Base(path)), ".gguf")
	plugin := &Yzma{ModelsDir: filepath.Dir(path), LibPath: lib}
	g := genkit.Init(t.Context(), genkit.WithPlugins(plugin))
	defer func() {
		if err := plugin.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	}()
	if names := discoveryNames(t.Context(), plugin); !slices.Contains(names, api.NewName(provider, name)) {
		t.Fatalf("native catalog %v does not contain %q", names, name)
	}
	model := Model(g, name)
	if model == nil {
		t.Fatal("discovered model did not resolve")
	}
	var streamed strings.Builder
	resp, err := model.Generate(t.Context(), ai.NewModelRequest(
		&GenerateContentConfig{Temperature: ptr(0.0), MaxOutputTokens: ptr(16)},
		ai.NewUserTextMessage("Once upon a time"),
	), func(_ context.Context, chunk *ai.ModelResponseChunk) error {
		for _, part := range chunk.Content {
			streamed.WriteString(part.Text)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() == "" || resp.Text() != streamed.String() || resp.Usage == nil ||
		resp.Usage.InputTokens < 1 || resp.Usage.OutputTokens < 1 || resp.Usage.OutputTokens > 16 {
		t.Errorf("invalid native response: %+v, streamed=%q", resp, streamed.String())
	}
}
