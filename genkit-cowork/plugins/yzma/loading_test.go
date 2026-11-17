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
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/hybridgroup/yzma/pkg/llama"
)

type testModelOperator struct {
	init func(context.Context, string) error
	load func(context.Context, ModelOptions) (llama.Model, error)
	free func(context.Context, llama.Model) error
}

func (o *testModelOperator) Init(ctx context.Context, path string) error {
	if o.init != nil {
		return o.init(ctx, path)
	}
	return nil
}

func (o *testModelOperator) LoadModel(ctx context.Context, opts ModelOptions) (llama.Model, error) {
	return o.load(ctx, opts)
}

func (o *testModelOperator) FreeModel(ctx context.Context, model llama.Model) error {
	if o.free != nil {
		return o.free(ctx, model)
	}
	return nil
}

func initLoadingTest(t *testing.T, operator ModelOperator) (*Yzma, *genkit.Genkit) {
	t.Helper()
	plugin := &Yzma{LibPath: t.TempDir(), Operator: operator}
	g := genkit.Init(t.Context(), genkit.WithPlugins(plugin))
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.WithoutCancel(t.Context()), 5*time.Second)
		defer cancel()
		if err := plugin.Close(ctx); err != nil {
			t.Errorf("cleanup: %v", err)
		}
	})
	return plugin, g
}

func testModelDefinition() ModelDefinition {
	return ModelDefinition{
		Name: "local-chat", Type: "chat",
		Loading: ModelOptions{GGUFPath: "chat.gguf"},
	}
}

func TestDefineModelLazyLoading(t *testing.T) {
	var initializations, loads, frees int
	op := &testModelOperator{
		init: func(context.Context, string) error {
			initializations++
			return nil
		},
		load: func(ctx context.Context, opts ModelOptions) (llama.Model, error) {
			loads++
			if opts.GGUFPath != "chat.gguf" {
				t.Errorf("path = %q, want chat.gguf", opts.GGUFPath)
			}
			return 42, nil
		},
		free: func(ctx context.Context, model llama.Model) error {
			frees++
			if model != 42 {
				t.Errorf("freed handle = %d, want 42", model)
			}
			return nil
		},
	}
	plugin, g := initLoadingTest(t, op)
	model := plugin.DefineModel(g, testModelDefinition(), nil)
	if initializations != 1 || loads != 0 || Model(g, "local-chat") == nil {
		t.Fatalf("registration: initializations=%d loads=%d; want initialized and registered without loading",
			initializations, loads)
	}
	_, err := model.Generate(t.Context(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("Hello")},
	}, nil)
	if !errors.Is(err, ErrGenerationNotImplemented) {
		t.Fatalf("Generate error = %v, want ErrGenerationNotImplemented", err)
	}
	if err := plugin.Preload(t.Context(), "local-chat"); err != nil {
		t.Fatal(err)
	}
	if loads != 1 {
		t.Fatalf("loads = %d, want 1", loads)
	}
	for range 2 {
		if err := plugin.Close(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	if frees != 1 {
		t.Errorf("frees = %d, want 1", frees)
	}
	_, err = model.Generate(t.Context(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("Hello again")},
	}, nil)
	if !errors.Is(err, ErrClosed) {
		t.Errorf("Generate after Close = %v, want ErrClosed", err)
	}
	if err := plugin.Preload(t.Context(), "local-chat"); !errors.Is(err, ErrClosed) {
		t.Errorf("Preload after Close = %v, want ErrClosed", err)
	}
}

func TestConcurrentPreload(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	var loads atomic.Int32
	started, finish := make(chan struct{}), make(chan struct{})
	op := &testModelOperator{
		load: func(ctx context.Context, opts ModelOptions) (llama.Model, error) {
			if loads.Add(1) == 1 {
				close(started)
			}
			select {
			case <-ctx.Done():
				return 0, ctx.Err()
			case <-finish:
				return 42, nil
			}
		},
	}
	plugin, g := initLoadingTest(t, op)
	plugin.DefineModel(g, testModelDefinition(), nil)
	const callers = 8
	results := make(chan error, callers)
	for range callers {
		go func() { results <- plugin.Preload(ctx, "local-chat") }()
	}
	select {
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	case <-started:
	}
	// A waiter can cancel independently of the caller performing the load.
	waitCtx, stopWaiting := context.WithCancel(ctx)
	waitResult := make(chan error, 1)
	go func() { waitResult <- plugin.Preload(waitCtx, "local-chat") }()
	stopWaiting()
	if err := <-waitResult; !errors.Is(err, context.Canceled) {
		t.Errorf("canceled waiter = %v, want context.Canceled", err)
	}
	close(finish)
	for range callers {
		select {
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case err := <-results:
			if err != nil {
				t.Error(err)
			}
		}
	}
	if got := loads.Load(); got != 1 {
		t.Errorf("loads = %d, want 1", got)
	}
}

func TestPreloadRetriesFailures(t *testing.T) {
	loadErr := errors.New("not enough memory")
	var loads int
	op := &testModelOperator{
		load: func(context.Context, ModelOptions) (llama.Model, error) {
			loads++
			switch loads {
			case 1:
				return 0, loadErr
			case 2:
				return 0, nil
			default:
				return 42, nil
			}
		},
	}
	plugin, g := initLoadingTest(t, op)
	plugin.DefineModel(g, testModelDefinition(), nil)
	if err := plugin.Preload(t.Context(), "local-chat"); !errors.Is(err, loadErr) {
		t.Fatalf("first load = %v, want original load error", err)
	}
	if err := plugin.Preload(t.Context(), "local-chat"); err == nil {
		t.Fatal("zero handle must not be cached as a successful load")
	}
	if err := plugin.Preload(t.Context(), "local-chat"); err != nil {
		t.Fatal(err)
	}
	if loads != 3 {
		t.Errorf("loads = %d, want 3", loads)
	}
}

func TestCloseWaitsForModelUse(t *testing.T) {
	var frees int
	op := &testModelOperator{
		load: func(context.Context, ModelOptions) (llama.Model, error) { return 42, nil },
		free: func(context.Context, llama.Model) error {
			frees++
			return nil
		},
	}
	plugin, g := initLoadingTest(t, op)
	plugin.DefineModel(g, testModelDefinition(), nil)
	gen := plugin.models["local-chat"]
	if err := gen.acquire(t.Context()); err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	err := plugin.Close(canceled)
	if !errors.Is(err, context.Canceled) || frees != 0 {
		gen.release()
		t.Fatalf("Close while reserved = %v, frees=%d; want cancellation and no free", err, frees)
	}
	gen.release()
	if err := plugin.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if frees != 1 {
		t.Errorf("frees = %d, want 1", frees)
	}
}

func TestCloseDuringLoading(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	started, finish := make(chan struct{}), make(chan struct{})
	var frees int
	op := &testModelOperator{
		load: func(ctx context.Context, opts ModelOptions) (llama.Model, error) {
			close(started)
			select {
			case <-ctx.Done():
				return 0, ctx.Err()
			case <-finish:
				return 42, nil
			}
		},
		free: func(context.Context, llama.Model) error { frees++; return nil },
	}
	plugin, g := initLoadingTest(t, op)
	plugin.DefineModel(g, testModelDefinition(), nil)
	result := make(chan error, 1)
	go func() { result <- plugin.Preload(ctx, "local-chat") }()
	select {
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	case <-started:
	}
	canceled, stop := context.WithCancel(ctx)
	stop()
	if err := plugin.Close(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("Close = %v, want context.Canceled", err)
	}
	close(finish)
	if err := <-result; !errors.Is(err, ErrClosed) {
		t.Errorf("late load = %v, want ErrClosed", err)
	}
	if err := plugin.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if frees != 1 {
		t.Errorf("late-loaded model freed %d times, want 1", frees)
	}
}

func TestCloseRetriesFreeFailure(t *testing.T) {
	freeErr := errors.New("cleanup failed")
	var frees int
	op := &testModelOperator{
		load: func(context.Context, ModelOptions) (llama.Model, error) { return 42, nil },
		free: func(context.Context, llama.Model) error {
			frees++
			if frees == 1 {
				return freeErr
			}
			return nil
		},
	}
	plugin, g := initLoadingTest(t, op)
	plugin.DefineModel(g, testModelDefinition(), nil)
	if err := plugin.Preload(t.Context(), "local-chat"); err != nil {
		t.Fatal(err)
	}
	if err := plugin.Close(t.Context()); !errors.Is(err, freeErr) {
		t.Fatalf("first Close = %v, want original cleanup error", err)
	}
	if err := plugin.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if frees != 2 {
		t.Errorf("frees = %d, want 2", frees)
	}
}

func TestDefineModelSnapshotsConfiguration(t *testing.T) {
	op := &testModelOperator{load: func(ctx context.Context, opts ModelOptions) (llama.Model, error) {
		if *opts.NGPULayers != 0 || opts.DeviceSelection[0] != "GPU0" || opts.TensorSplit[0] != 1 {
			t.Errorf("loading options changed after registration: %+v", opts)
		}
		return 42, nil
	}}
	plugin, g := initLoadingTest(t, op)
	model := testModelDefinition()
	model.Loading.NGPULayers = ptr(int32(0))
	model.Loading.DeviceSelection = []string{"GPU0"}
	model.Loading.TensorSplit = []float32{1}
	model.Context.ContextSize = ptr(uint32(4096))
	model.GenerationDefaults = &GenerateContentConfig{
		Temperature: ptr(0.7), StopSequences: ptr([]string{"STOP"}),
	}
	plugin.DefineModel(g, model, nil)
	*model.Loading.NGPULayers = 20
	model.Loading.DeviceSelection[0] = "GPU1"
	model.Loading.TensorSplit[0] = 2
	*model.Context.ContextSize = 8192
	*model.GenerationDefaults.Temperature = 0.1
	(*model.GenerationDefaults.StopSequences)[0] = "changed"
	if err := plugin.Preload(t.Context(), model.Name); err != nil {
		t.Fatal(err)
	}
	snapshot := plugin.models[model.Name].model
	if *snapshot.Context.ContextSize != 4096 || *snapshot.GenerationDefaults.Temperature != 0.7 ||
		(*snapshot.GenerationDefaults.StopSequences)[0] != "STOP" {
		t.Error("context or generation defaults changed after registration")
	}
}

func TestDefineModelRejectsDuplicateRegistration(t *testing.T) {
	var loads int
	op := &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) {
		loads++
		return 42, nil
	}}
	plugin, g := initLoadingTest(t, op)
	model := testModelDefinition()
	plugin.DefineModel(g, model, nil)
	// A second registry must not replace the entry responsible for cleanup.
	other := genkit.Init(t.Context())
	func() {
		defer func() {
			if recover() == nil {
				t.Error("duplicate registration must panic")
			}
		}()
		plugin.DefineModel(other, model, nil)
	}()
	if loads != 0 || Model(other, model.Name) != nil {
		t.Error("duplicate registration loaded weights or changed the other registry")
	}
	if err := plugin.Preload(t.Context(), model.Name); err != nil {
		t.Fatal(err)
	}
}

// TestNativeModelLoading is opt-in because native libraries and a GGUF file are
// external dependencies. It does not download models or run token generation.
func TestNativeModelLoading(t *testing.T) {
	path, lib := os.Getenv("YZMA_TEST_MODEL"), os.Getenv("YZMA_LIB")
	if path == "" || lib == "" {
		t.Skip("set YZMA_TEST_MODEL and YZMA_LIB to exercise native model loading")
	}
	plugin := &Yzma{LibPath: lib}
	g := genkit.Init(t.Context(), genkit.WithPlugins(plugin))
	t.Cleanup(func() {
		if err := plugin.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	model := testModelDefinition()
	model.Loading.GGUFPath, model.Loading.NGPULayers = path, ptr(int32(0))
	plugin.DefineModel(g, model, nil)
	if err := plugin.Preload(t.Context(), model.Name); err != nil {
		t.Fatal(err)
	}
	if err := plugin.Preload(t.Context(), model.Name); err != nil {
		t.Fatal(err)
	}
}
