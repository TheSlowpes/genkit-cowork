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
	"strconv"
	"strings"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/message"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

func (defaultModelOperator) LoadProjector(ctx context.Context, model llama.Model, def ModelDefinition) (ProjectorHandle, error) {
	if llama.ModelHasEncoder(model) || !llama.ModelHasDecoder(model) {
		return 0, fmt.Errorf("media input requires a decoder-only model")
	}
	decoder, err := readGGUFHeader(ctx, def.Loading.GGUFPath)
	if err != nil {
		return 0, fmt.Errorf("read decoder metadata: %w", err)
	}
	projector, err := readGGUFHeader(ctx, def.Projector.GGUFPath)
	if err != nil {
		return 0, fmt.Errorf("read projector metadata: %w", err)
	}
	if def.Projector.Images {
		if err := validateVisionPair(decoder.values, projector); err != nil {
			return 0, err
		}
	}
	if err := loadMediaLibrary(ctx); err != nil {
		return 0, err
	}
	params := mtmd.ContextParamsDefault()
	params.Warmup = false
	params.PrintTimings = false
	params.UseGPU = def.Loading.NGPULayers == nil || *def.Loading.NGPULayers != 0
	if threads := def.Context.BatchThreads; threads != nil && *threads > 0 {
		params.Threads = *threads
	}
	params.SetProgressCallback(func(float32, uintptr) bool { return ctx.Err() == nil })
	defer params.SetProgressCallback(nil)
	mctx, err := mtmd.InitFromFile(def.Projector.GGUFPath, model, params)
	err = errors.Join(err, ctx.Err())
	if err == nil && def.Projector.Images && !mtmd.SupportVision(mctx) {
		err = fmt.Errorf("loaded projector does not support image input")
	}
	return ProjectorHandle(mctx), err
}

func (defaultModelOperator) FreeProjector(ctx context.Context, projector ProjectorHandle) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return mtmd.Free(mtmd.Context(projector))
}

// The stock SmolVLM template takes Transformers content arrays and emits its
// own image tokens. mtmd supplies those tokens instead, using the ordered slots
// in this equivalent string-content template.
const smolVLMTemplate = `<|im_start|>{% for message in messages %}{{ message['role'] | capitalize }}: {{ message['content'] }}<end_of_utterance>
{% endfor %}Assistant:`

func (defaultModelOperator) GenerateMedia(ctx context.Context, model llama.Model, projector ProjectorHandle, input MediaGenerationInput, onText func(context.Context, string) error) (result TextGenerationResult, err error) {
	if err := ctx.Err(); err != nil {
		return result, err
	}
	mctx := mtmd.Context(projector)
	vocab := llama.ModelGetVocab(model)
	input.bosToken = tokenSpelling(vocab, llama.VocabBOS(vocab))
	input.eosToken = tokenSpelling(vocab, llama.VocabEOS(vocab))
	prompt, payloads, err := renderMediaPrompt(input, func() string {
		if input.ChatTemplateName == "" {
			if name, _ := llama.ModelMetaValStr(model, "general.basename"); name == "SmolVLM" {
				return smolVLMTemplate
			}
		}
		return embeddedChatTemplate(model, input.ChatTemplateName, false)
	})
	if err != nil {
		return result, err
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	params, err := resolveContextParams(input.Context)
	if err != nil {
		return result, err
	}
	lctx, err := llama.InitFromModel(model, params)
	if err != nil {
		return result, fmt.Errorf("allocate inference context: %w", err)
	}
	llama.SetAbortCallback(lctx, func() bool { return ctx.Err() != nil })
	defer func() {
		llama.SetAbortCallback(lctx, nil)
		err = errors.Join(err, llama.Free(lctx))
	}()
	sampler := textSampler(model, input.Sampler)
	if sampler == 0 {
		return result, fmt.Errorf("allocate sampler")
	}
	defer llama.SamplerFree(sampler)
	addSpecial := input.bosToken == "" || !strings.HasPrefix(prompt[0], input.bosToken)
	var position llama.Pos
	inputTokens, remaining, err := prefillMedia(ctx, lctx, mctx, sampler, prompt, payloads, addSpecial, &position)
	if err != nil {
		return result, err
	}
	session := nativeTokenSession{ctx: lctx, vocab: vocab, sampler: sampler, position: &position}
	return sampleTokens(ctx, session, inputTokens, remaining, input.TextGenerationInput, onText)
}

// renderMediaPrompt lets the chat template place turns and special tokens but
// rejects templates that drop, duplicate, or reorder media slots.
func renderMediaPrompt(input MediaGenerationInput, template func() string) ([]string, []MediaInputPart, error) {
	var payloads []MediaInputPart
	input.TextGenerationInput.Messages = make([]message.Message, 0, len(input.Messages))
	for _, msg := range input.Messages {
		var text strings.Builder
		for _, part := range msg.Parts {
			if part.MIMEType != "" {
				text.WriteString(mediaSlot(len(payloads)))
				payloads = append(payloads, part)
			} else {
				text.WriteString(part.Text)
			}
		}
		input.TextGenerationInput.Messages = append(input.TextGenerationInput.Messages, message.Chat{Role: msg.Role, Content: text.String()})
	}
	rendered, _, err := prepareTextPrompt(input.TextGenerationInput, template, llama.ChatApplyTemplate)
	if err != nil {
		return nil, nil, err
	}
	spans := make([]string, 0, len(payloads)+1)
	for i := range payloads {
		prefix, _, _ := strings.Cut(rendered, mediaSlotPrefix)
		suffix, ok := strings.CutPrefix(rendered[len(prefix):], mediaSlot(i))
		if !ok {
			return nil, nil, fmt.Errorf("chat template changed the ordered media slots")
		}
		spans = append(spans, prefix)
		rendered = suffix
	}
	if strings.Contains(rendered, mediaSlotPrefix) {
		return nil, nil, fmt.Errorf("chat template changed the ordered media slots")
	}
	return append(spans, rendered), payloads, nil
}

func mediaSlot(index int) string {
	return mediaSlotPrefix + strconv.Itoa(index) + "|>"
}

func prefillMedia(ctx context.Context, lctx llama.Context, mctx mtmd.Context, sampler llama.Sampler, spans []string, payloads []MediaInputPart, addSpecial bool, position *llama.Pos) (int, int, error) {
	var bitmaps []mtmd.Bitmap
	defer func() {
		for _, bitmap := range bitmaps {
			mtmd.BitmapFree(bitmap)
		}
	}()
	chunks := mtmd.InputChunksInit()
	if chunks == 0 {
		return 0, 0, fmt.Errorf("allocate media chunks")
	}
	defer mtmd.InputChunksFree(chunks)
	parts := make([]*mtmd.InputPart, 0, len(spans)+len(payloads))
	for i, span := range spans {
		if err := ctx.Err(); err != nil {
			return 0, 0, err
		}
		if span != "" {
			parts = append(parts, mtmd.NewInputTextPart(mtmd.NewInputText(span, false, true)))
		}
		if i < len(payloads) {
			bitmap, err := mediaBitmap(ctx, payloads[i])
			if err != nil {
				return 0, 0, err
			}
			bitmaps = append(bitmaps, bitmap)
			parts = append(parts, mtmd.NewInputBitmapPart(bitmap))
		}
	}
	if code := mtmd.TokenizeFromParts(mctx, chunks, parts, addSpecial); code != 0 {
		return 0, 0, fmt.Errorf("tokenize media prompt: mtmd status %d", code)
	}
	capacity, batch := int(llama.NCtxSeq(lctx)), int32(llama.NBatch(lctx))
	var tokens, positions int
	for i := range mtmd.InputChunksSize(chunks) {
		chunk := mtmd.InputChunksGet(chunks, i)
		tokens += int(mtmd.InputChunkGetNTokens(chunk))
		positions += int(mtmd.InputChunkGetNPos(chunk))
		if mtmd.InputChunkGetType(chunk) == mtmd.InputChunkTypeText {
			for _, token := range mtmd.InputChunkGetTokensText(chunk) {
				llama.SamplerAccept(sampler, token)
			}
		}
	}
	if tokens <= 0 || max(tokens, positions) >= capacity || batch <= 0 {
		return 0, 0, fmt.Errorf("media prompt has %d tokens at %d positions; context capacity is %d and must leave room for output", tokens, positions, capacity)
	}
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	code := mtmd.HelperEvalChunks(mctx, lctx, chunks, 0, 0, batch, true, position)
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	if code != 0 {
		return 0, 0, fmt.Errorf("evaluate media prompt: mtmd status %d", code)
	}
	remaining := capacity - max(tokens, int(*position))
	if remaining <= 0 {
		return 0, 0, fmt.Errorf("media prompt leaves no room for output")
	}
	return tokens, remaining, nil
}

func mediaBitmap(ctx context.Context, part MediaInputPart) (mtmd.Bitmap, error) {
	switch part.MIMEType {
	case "image/jpeg", "image/png":
		return imageBitmap(ctx, part.image)
	default:
		return 0, fmt.Errorf("cannot create bitmap for media MIME type %q", part.MIMEType)
	}
}
