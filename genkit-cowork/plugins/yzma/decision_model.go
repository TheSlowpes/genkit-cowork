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
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ardanlabs/jinja"
	"github.com/hybridgroup/yzma/exp/decide"
)

// DecisionReadout selects how Yzma reads scores from a System One model.
type DecisionReadout string

const (
	// DecisionReadoutGGUF uses the embedded System One metadata and template.
	DecisionReadoutGGUF DecisionReadout = "gguf"
	// DecisionReadoutJevStyle requires a matching readout_config.json.
	DecisionReadoutJevStyle DecisionReadout = "jev-style"
	// DecisionReadoutJevK5 requires a matching jevk5_config.json.
	DecisionReadoutJevK5 DecisionReadout = "jevk5"
	// DecisionReadoutDecider requires a matching decider_config.json.
	DecisionReadoutDecider DecisionReadout = "decider"
)

// DecisionOptions selects the model's readout and its calibration sidecar.
// Leave ConfigPath empty for an embedded GGUF readout.
type DecisionOptions struct {
	Readout    DecisionReadout `json:"readout"`
	ConfigPath string          `json:"configPath,omitempty"`
}

// DecisionOperator is the optional decision capability of a ModelOperator.
// The returned model owns its weights and context. On a load error, the
// operator must free any partial allocation and return a nil model.
type DecisionOperator interface {
	LoadDecision(ctx context.Context, path string, options DecisionOptions, params ContextOptions) (DecisionModel, error)
}

// DecisionModel scores a batch about one text or JSON state. The plugin
// serializes calls to one model and calls Close when it no longer needs it.
// On a Close error the model must remain valid so cleanup can be retried.
type DecisionModel interface {
	DecideMany(ctx context.Context, state any, questions []decide.Question) ([]*decide.Result, error)
	Close(ctx context.Context) error
}

func validateDecisionDefinition(model ModelDefinition) error {
	o := model.Decision
	if model.Type != ModelTypeDecision {
		if o != (DecisionOptions{}) {
			return errors.New("decision options require a decision model")
		}
		return nil
	}
	if model.ChatTemplate != "" || model.ChatTemplateName != "" || model.ToolCalling ||
		model.GenerationDefaults != nil {
		return errors.New("decision models cannot use chat templates, tools, or generation defaults")
	}
	l := model.Loading
	if l.NGPULayers != nil || len(l.DeviceSelection) > 0 || l.GPUSplitMode != "" ||
		l.MainGPU != nil || len(l.TensorSplit) > 0 || l.LoadMode != "" || l.LazyMode != "" {
		return errors.New("decision models use native weight loading defaults")
	}
	if strings.TrimSpace(model.Loading.GGUFPath) == "" {
		return errors.New("decision models require a GGUF path")
	}
	if model.Context.LogicalMaximumBatchSize != nil || model.Context.BatchThreads != nil {
		return errors.New("decision models set logical batch size and batch threads internally")
	}
	switch o.Readout {
	case DecisionReadoutGGUF:
		if o.ConfigPath != "" {
			return errors.New("embedded decision readouts do not use a sidecar")
		}
	case DecisionReadoutJevStyle, DecisionReadoutJevK5, DecisionReadoutDecider:
		if strings.TrimSpace(o.ConfigPath) == "" {
			return fmt.Errorf("decision readout %q requires a calibration sidecar path", o.Readout)
		}
	default:
		return fmt.Errorf("unsupported decision readout %q", o.Readout)
	}
	return nil
}

const systemOneTemplateKey = "tokenizer.chat_template.systemone"

// discoverDecision claims a candidate before text classification if it has a
// System One marker or a per-file calibration sidecar. A broken decision
// artifact must not fall through and accidentally be listed as a text model.
func discoverDecision(file ggufCandidate) (DecisionOptions, bool, error) {
	values := file.header.values
	arch := values["general.architecture"].text
	kind, hasKind := values[arch+".decision.type"]
	template, hasTemplate := values[systemOneTemplateKey]
	var o DecisionOptions
	if hasKind || hasTemplate {
		o.Readout = DecisionReadoutGGUF
	}
	stem := strings.TrimSuffix(file.path, filepath.Ext(file.path))
	for _, sidecar := range []struct {
		suffix  string
		readout DecisionReadout
	}{
		{".readout.json", DecisionReadoutJevStyle},
		{".jevk5.json", DecisionReadoutJevK5},
		{".decider.json", DecisionReadoutDecider},
	} {
		path := stem + sidecar.suffix
		if _, err := os.Lstat(path); errors.Is(err, os.ErrNotExist) {
			continue
		} else if err != nil {
			return o, true, fmt.Errorf("inspect decision sidecar: %w", err)
		}
		if o.Readout != "" {
			return o, true, errors.New("ambiguous decision readout; choose one with DefineModel")
		}
		o = DecisionOptions{Readout: sidecar.readout, ConfigPath: path}
	}
	if o.Readout == "" {
		return o, false, nil
	}
	_, imageProjector := values["clip.projector_type"]
	_, speechDecoder := values["clip.gen.audio.projector_type"]
	if arch == "" || arch == "clip" || imageProjector || speechDecoder ||
		values["general.type"].text != "model" || len(file.header.tensors) == 0 {
		return o, true, errors.New("decision readout requires a model GGUF")
	}
	if o.Readout == DecisionReadoutGGUF {
		// Match the embedded kinds understood by pinned Yzma exp/decide.
		switch kind.text {
		case "openjev", "lev", "laya", "kev", "nimble", "pplx-decider",
			"lfm2-d1", "lfm2-d1-omni", "lfm3-d1", "lfm3-d1-omni",
			"granite-switch", "h2o-lightning", "clef", "vllm-decision2", "vllm-d3":
		default:
			return o, true, fmt.Errorf("unsupported or missing embedded decision readout %q", kind.text)
		}
		if strings.TrimSpace(template.text) == "" || strings.ContainsRune(template.text, '\x00') {
			return o, true, errors.New("decision model has no valid systemone template")
		}
		if _, err := jinja.Compile(template.text); err != nil {
			return o, true, fmt.Errorf("invalid systemone template: %w", err)
		}
	} else if _, err := readDecisionSidecar(o); err != nil {
		return o, true, err
	}
	return o, true, nil
}

// decisionCalibration holds the parsed config for exactly one sidecar readout.
type decisionCalibration struct {
	jev     *decide.Config
	k5      *decide.JevK5Config
	decider *decide.DeciderConfig
}

func readDecisionSidecar(o DecisionOptions) (c decisionCalibration, err error) {
	const limit = 64 * 1024
	info, err := os.Stat(o.ConfigPath)
	if err != nil {
		return c, fmt.Errorf("decision sidecar: %w", err)
	}
	if !info.Mode().IsRegular() || info.Size() > limit {
		return c, errors.New("decision sidecar must be a regular file no larger than 64 KiB")
	}
	f, err := os.Open(o.ConfigPath)
	if err != nil {
		return c, fmt.Errorf("decision sidecar: %w", err)
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return c, fmt.Errorf("decision sidecar: %w", err)
	}
	if len(data) > limit {
		return c, errors.New("decision sidecar exceeds 64 KiB")
	}
	if trimmed := bytes.TrimSpace(data); len(trimmed) == 0 || trimmed[0] != '{' {
		return c, errors.New("decision sidecar must contain a JSON object")
	}
	switch o.Readout {
	case DecisionReadoutJevStyle:
		c.jev, err = decide.ParseConfig(data)
	case DecisionReadoutJevK5:
		c.k5, err = decide.ParseJevK5Config(data)
	case DecisionReadoutDecider:
		c.decider, err = decide.ParseDeciderConfig(data)
	}
	if err != nil {
		return c, fmt.Errorf("invalid %s sidecar %q: %w", o.Readout, o.ConfigPath, err)
	}
	return c, nil
}

func (g *generator) loadDecision(ctx context.Context) error {
	operator, ok := g.operator.(DecisionOperator)
	if !ok {
		return errors.New("yzma: configured operator does not support decision models")
	}
	loaded, err := operator.LoadDecision(ctx, g.model.Loading.GGUFPath, g.model.Decision, g.model.Context)
	if err != nil {
		return err
	}
	if loaded == nil {
		return errors.New("decision operator returned a nil model")
	}
	g.decision = loaded
	return nil
}

type nativeDecision struct{ decider *decide.Decider }

func (defaultModelOperator) LoadDecision(ctx context.Context, path string, options DecisionOptions, params ContextOptions) (DecisionModel, error) {
	var opts decide.Options
	if params.ContextSize != nil {
		opts.ContextSize = *params.ContextSize
	}
	if params.PhysicalMaximumBatchSize != nil {
		opts.UBatch = *params.PhysicalMaximumBatchSize
	}
	if params.Threads != nil {
		opts.Threads = *params.Threads
	}
	var c decisionCalibration
	var err error
	if options.Readout != DecisionReadoutGGUF {
		c, err = readDecisionSidecar(options)
		if err != nil {
			return nil, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var d *decide.Decider
	switch options.Readout {
	case DecisionReadoutGGUF:
		d, err = decide.Open(path, opts)
	case DecisionReadoutJevStyle:
		d, err = decide.NewFromConfig(path, c.jev, opts)
	case DecisionReadoutJevK5:
		d, err = decide.NewJevK5FromConfig(path, c.k5, opts)
	case DecisionReadoutDecider:
		d, err = decide.NewDeciderModelFromConfig(path, c.decider, opts)
	default:
		return nil, fmt.Errorf("unsupported decision readout %q", options.Readout)
	}
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		d.Close()
		return nil, err
	}
	return &nativeDecision{decider: d}, nil
}

func (d *nativeDecision) DecideMany(ctx context.Context, state any, questions []decide.Question) ([]*decide.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	results, err := d.decider.DecideMany(state, questions, "")
	return results, errors.Join(err, ctx.Err())
}

func (d *nativeDecision) Close(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	d.decider.Close()
	return nil
}
