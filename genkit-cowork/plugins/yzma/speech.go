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
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/firebase/genkit/go/ai"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

const (
	speechMIME       = "audio/wav"
	speechSampleRate = 32000
	maxSpeechBytes   = 1024
	maxSpeechFrames  = 512
)

// SpeechConfig configures a Soprano speech request through ai.WithConfig.
// An omitted voice or "default" uses the model's fixed voice. Other values
// are rejected; the plugin does not accept reference audio or clone voices.
type SpeechConfig struct {
	Voice string `json:"voice,omitempty"`
}

// SpeechOperator extends ModelOperator with buffered speech synthesis.
// A decoder belongs to the supplied backbone model and is reused until Close.
// Calls for a single registration are serialized; distinct models may run concurrently.
// LoadSpeechDecoder must clean up on error and return a nonzero handle on
// success. FreeSpeechDecoder must leave the handle valid if cleanup fails.
// GenerateSpeech must free its request resources before returning. It returns
// a complete canonical RIFF WAV, mono signed 16-bit PCM at 32 kHz.
type SpeechOperator interface {
	LoadSpeechDecoder(ctx context.Context, model llama.Model, path string, opts ModelOptions, inference ContextOptions) (mtmd.Context, error)
	FreeSpeechDecoder(ctx context.Context, decoder mtmd.Context) error
	GenerateSpeech(ctx context.Context, model llama.Model, decoder mtmd.Context, inference ContextOptions, text string) ([]byte, error)
}

func validateSpeechDefinition(model ModelDefinition, operator ModelOperator) error {
	if model.Type != ModelTypeSpeech {
		if model.SpeechDecoderPath != "" {
			return fmt.Errorf("SpeechDecoderPath requires a speech model")
		}
		return nil
	}
	if strings.TrimSpace(model.SpeechDecoderPath) == "" || strings.ContainsRune(model.SpeechDecoderPath, '\x00') {
		return fmt.Errorf("speech models require a non-empty SpeechDecoderPath without NUL bytes")
	}
	if model.ChatTemplate != "" || model.ChatTemplateName != "" || model.ToolCalling || model.GenerationDefaults != nil {
		return fmt.Errorf("speech models do not support chat templates, tools, or text generation defaults")
	}
	if _, ok := operator.(SpeechOperator); !ok {
		return fmt.Errorf("configured operator does not implement speech generation")
	}
	return nil
}

func (g *generator) loadSpeechDecoder(ctx context.Context) error {
	if err := g.available(ctx); err != nil {
		return err
	}
	op := g.operator.(SpeechOperator)
	decoder, err := op.LoadSpeechDecoder(ctx, g.loaded, g.model.SpeechDecoderPath, g.model.Loading, g.model.Context)
	if err != nil {
		return fmt.Errorf("load speech decoder: %w", err)
	}
	if decoder == 0 {
		return fmt.Errorf("load speech decoder: operator returned a zero handle")
	}
	g.speechDecoder = decoder
	return g.available(ctx)
}

func (g *generator) generateSpeech(ctx context.Context, input *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
	if cb != nil {
		return nil, fmt.Errorf("yzma: speech output does not support streaming")
	}
	text, err := speechPrompt(input)
	if err != nil {
		return nil, err
	}
	if err := g.acquire(ctx); err != nil {
		return nil, err
	}
	defer g.release()
	wav, err := g.operator.(SpeechOperator).GenerateSpeech(ctx, g.loaded, g.speechDecoder, g.model.Context, text)
	if err = errors.Join(err, ctx.Err()); err != nil {
		return nil, fmt.Errorf("generate speech with model %q: %w", g.model.Name, err)
	}
	if !validSpeechWAV(wav) {
		return nil, fmt.Errorf("yzma: speech operator did not return a complete 32 kHz mono PCM16 WAV")
	}
	media := ai.NewMediaPart(speechMIME, "data:"+speechMIME+";base64,"+base64.StdEncoding.EncodeToString(wav))
	return &ai.ModelResponse{
		Message: ai.NewModelMessage(media), FinishReason: ai.FinishReasonStop, Request: input,
	}, nil
}

func speechPrompt(input *ai.ModelRequest) (string, error) {
	if input == nil || len(input.Messages) != 1 || input.Messages[0] == nil || input.Messages[0].Role != ai.RoleUser {
		return "", fmt.Errorf("yzma: speech requires exactly one user text message")
	}
	if len(input.Tools) > 0 || len(input.Docs) > 0 {
		return "", fmt.Errorf("yzma: speech does not support tools or document grounding")
	}
	if out := input.Output; out != nil {
		// Genkit v1.9 adds text/plain with no Format when Generate has no
		// explicit formatter. It does not have a built-in media formatter.
		validFormat := out.Format == "" || out.Format == ai.OutputFormatMedia
		validType := out.ContentType == "" || out.ContentType == speechMIME ||
			(out.Format == "" && out.ContentType == "text/plain")
		if !validFormat || !validType || out.Constrained || len(out.Schema) != 0 {
			return "", fmt.Errorf("yzma: speech only supports WAV output")
		}
	}
	if err := validateSpeechConfig(input.Config); err != nil {
		return "", err
	}
	var text strings.Builder
	for _, part := range input.Messages[0].Content {
		if part == nil || !part.IsText() || part.Metadata["purpose"] == "output" {
			return "", fmt.Errorf("yzma: speech only accepts plain text input")
		}
		if len(part.Text) > maxSpeechBytes-text.Len() {
			return "", fmt.Errorf("yzma: speech prompt exceeds %d bytes", maxSpeechBytes)
		}
		text.WriteString(part.Text)
	}
	s := text.String()
	if strings.TrimSpace(s) == "" || !utf8.ValidString(s) || strings.ContainsRune(s, '\x00') {
		return "", fmt.Errorf("yzma: speech requires non-empty UTF-8 text without NUL bytes")
	}
	return s, nil
}

func validateSpeechConfig(value any) error {
	var voice string
	switch cfg := value.(type) {
	case nil:
	case SpeechConfig:
		voice = cfg.Voice
	case *SpeechConfig:
		if cfg != nil {
			voice = cfg.Voice
		}
	case map[string]any:
		for key, val := range cfg {
			if key != "voice" {
				return fmt.Errorf("yzma: unknown speech config field %q", key)
			}
			var ok bool
			if voice, ok = val.(string); !ok {
				return fmt.Errorf("yzma: speech voice must be a string")
			}
		}
	default:
		return fmt.Errorf("yzma: unsupported config type %T; use SpeechConfig", value)
	}
	if voice != "" && voice != "default" {
		return fmt.Errorf("yzma: unsupported speech voice %q; Soprano only supports %q", voice, "default")
	}
	return nil
}

// validSpeechWAV checks the canonical 44-byte header produced by libmtmd.
func validSpeechWAV(wav []byte) bool {
	const headerSize = 44
	le := binary.LittleEndian
	return len(wav) > headerSize && len(wav)%2 == 0 &&
		string(wav[:4]) == "RIFF" && uint64(le.Uint32(wav[4:8])) == uint64(len(wav)-8) &&
		string(wav[8:16]) == "WAVEfmt " && le.Uint32(wav[16:20]) == 16 &&
		le.Uint16(wav[20:22]) == 1 && le.Uint16(wav[22:24]) == 1 &&
		le.Uint32(wav[24:28]) == speechSampleRate && le.Uint32(wav[28:32]) == speechSampleRate*2 &&
		le.Uint16(wav[32:34]) == 2 && le.Uint16(wav[34:36]) == 16 &&
		string(wav[36:40]) == "data" && uint64(le.Uint32(wav[40:44])) == uint64(len(wav)-headerSize)
}

func (defaultModelOperator) LoadSpeechDecoder(ctx context.Context, model llama.Model, path string, opts ModelOptions, inference ContextOptions) (mtmd.Context, error) {
	if err := loadMediaLibrary(ctx); err != nil {
		return 0, err
	}
	if _, err := mtmd.HelperGenAudioInputDefault(); err != nil {
		return 0, fmt.Errorf("speech requires libmtmd audio generation helpers from the Yzma v1.33.0 library build: %w", err)
	}
	if tokenizer, _ := llama.ModelMetaValStr(model, "tokenizer.ggml.pre"); tokenizer != "soprano" {
		return 0, fmt.Errorf("speech requires a Soprano backbone GGUF")
	}
	h, err := readGGUFHeader(ctx, path)
	if err != nil {
		return 0, err
	}
	if h.values["clip.gen.audio.projector_type"].text != "soprano" {
		return 0, fmt.Errorf("speech requires a matching Soprano mmproj decoder GGUF")
	}
	params := mtmd.ContextParamsDefault()
	params.Warmup = false
	if opts.NGPULayers != nil && *opts.NGPULayers == 0 {
		params.UseGPU = false
	}
	if inference.Threads != nil && *inference.Threads > 0 {
		params.Threads = *inference.Threads
	}
	params.SetProgressCallback(func(float32, uintptr) bool { return ctx.Err() == nil })
	defer params.SetProgressCallback(nil)
	decoder, err := mtmd.InitFromFile(path, model, params)
	if err = errors.Join(err, ctx.Err()); err == nil {
		caps, capErr := mtmd.HelperGetCapabilities(model, decoder)
		if capErr != nil {
			err = capErr
		} else if !caps.AudioGen {
			err = fmt.Errorf("Soprano GGUF pair does not support speech output")
		}
	}
	if err != nil && decoder != 0 {
		return 0, errors.Join(err, mtmd.Free(decoder))
	}
	return decoder, err
}

func (defaultModelOperator) FreeSpeechDecoder(ctx context.Context, decoder mtmd.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return mtmd.Free(decoder)
}

func (defaultModelOperator) GenerateSpeech(ctx context.Context, model llama.Model, decoder mtmd.Context, inference ContextOptions, text string) (wav []byte, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	params, err := resolveContextParams(inference)
	if err != nil {
		return nil, err
	}
	params.Embeddings = 1 // The speech helper reads the backbone hidden state.
	lctx, err := llama.InitFromModel(model, params)
	if err != nil {
		return nil, fmt.Errorf("allocate speech inference context: %w", err)
	}
	llama.SetAbortCallback(lctx, func() bool { return ctx.Err() != nil })
	defer func() {
		llama.SetAbortCallback(lctx, nil)
		err = errors.Join(err, llama.Free(lctx), ctx.Err())
	}()
	gen, err := mtmd.HelperGenAudioInit(lctx, decoder)
	if err != nil {
		return nil, err
	}
	defer mtmd.HelperGenAudioFree(gen)
	input, err := mtmd.HelperGenAudioInputDefault()
	if err != nil {
		return nil, err
	}
	input.Stream = false
	input.OutType = mtmd.HelperGenAudioOutTypeWAV
	input.SetPrompt(text)
	if err := mtmd.HelperGenAudioSetInput(gen, &input); err != nil {
		return nil, err
	}
	batchSize := int32(llama.NBatch(lctx))
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		left, err := mtmd.HelperGenAudioStepPrompt(gen, batchSize)
		if err != nil {
			return nil, err
		}
		if left == 0 {
			break
		}
	}
	if err := generateSpeechFrames(ctx, model, lctx, gen); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	output, err := mtmd.HelperGenAudioGetOutput(gen)
	if err != nil {
		return nil, err
	}
	if output.SampleRate != speechSampleRate || output.NSamples <= 0 {
		return nil, fmt.Errorf("speech decoder returned %d samples at %d Hz", output.NSamples, output.SampleRate)
	}
	return output.Data, nil
}

func generateSpeechFrames(ctx context.Context, model llama.Model, lctx llama.Context, gen mtmd.HelperGenAudio) error {
	state, err := llama.GetEmbeddingsIth(lctx, -1, llama.ModelNEmbd(model))
	if err != nil {
		return err
	}
	// Soprano uses greedy sampling. Its tokenizer includes reserved tokens
	// that the model suppresses during semantic code generation.
	vocab := llama.ModelGetVocab(model)
	var bias []llama.LogitBias
	for _, tok := range llama.VocabGetSuppressTokens(vocab) {
		bias = append(bias, llama.LogitBias{Token: tok, Bias: float32(math.Inf(-1))})
	}
	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	if sampler == 0 {
		return fmt.Errorf("allocate speech sampler")
	}
	defer llama.SamplerFree(sampler)
	if len(bias) > 0 {
		llama.SamplerChainAdd(sampler, llama.SamplerInitLogitBias(llama.VocabNTokens(vocab), int32(len(bias)), &bias[0]))
	}
	llama.SamplerChainAdd(sampler, llama.SamplerInitGreedy())
	for frames := 0; ; frames++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		token := llama.SamplerSample(sampler, lctx, -1)
		if llama.VocabIsEOG(vocab, token) {
			return nil
		}
		if frames == maxSpeechFrames {
			return fmt.Errorf("speech reached the %d-frame limit before end of speech; shorten the prompt", maxSpeechFrames)
		}
		next, stop, err := mtmd.HelperGenAudioStepGen(gen, token, state)
		if err != nil {
			return err
		}
		if stop || next == nil {
			return fmt.Errorf("speech stopped before end of speech; check the model's context capacity")
		}
		state = next
	}
}
