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
	"fmt"
	"image"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/hybridgroup/yzma/pkg/llama"
)

// ProjectorHandle identifies an operator-owned projector tied to one model.
type ProjectorHandle uintptr

// MediaGenerationOperator optionally extends ModelOperator with media inference.
// The plugin serializes access to a projector and frees it before its model.
// LoadProjector must return a nonzero handle on success. A nonzero handle
// returned with an error transfers ownership for cleanup, not inference.
// FreeProjector must leave the handle valid on error so Close can retry.
// GenerateMedia owns its request resources and releases them before returning.
// Its text callback and result follow the TextGenerationOperator contract.
type MediaGenerationOperator interface {
	LoadProjector(ctx context.Context, model llama.Model, definition ModelDefinition) (ProjectorHandle, error)
	FreeProjector(ctx context.Context, projector ProjectorHandle) error
	GenerateMedia(ctx context.Context, model llama.Model, projector ProjectorHandle, input MediaGenerationInput, onText func(context.Context, string) error) (TextGenerationResult, error)
}

// ConstrainedMediaGenerationOperator extends image chat with GBNF decoding.
// Apply the grammar's "root" rule to output tokens only, not the media prompt.
// Report token or context exhaustion as Limited and release request resources
// before returning, as with GenerateMedia.
type ConstrainedMediaGenerationOperator interface {
	MediaGenerationOperator
	GenerateConstrainedMedia(ctx context.Context, model llama.Model, projector ProjectorHandle, input MediaGenerationInput, grammar string) (TextGenerationResult, error)
}

// MediaGenerationInput contains generation settings and ordered chat messages.
// TextGenerationInput.Messages is unused. Read the ordered Messages instead.
type MediaGenerationInput struct {
	TextGenerationInput
	Messages []MediaMessage
}

// MediaMessage is a chat role and its original sequence of text and media.
type MediaMessage struct {
	Role  string
	Parts []MediaInputPart
}

// MediaInputPart is either text, or inline bytes with a validated MIME type.
type MediaInputPart struct {
	Text     string
	MIMEType string
	Data     []byte
	// VideoFrames holds the sampled frames of a validated video in display order.
	VideoFrames []image.Image
	image       image.Image
	pcm16       []byte
}

const (
	maxMediaParts   = 4
	maxRequestMedia = 8 << 20
	mediaSlotPrefix = "<|yzma-media-"
)

func validateProjectorOptions(model ModelDefinition) error {
	p := model.Projector
	if strings.TrimSpace(p.GGUFPath) == "" || strings.ContainsRune(p.GGUFPath, '\x00') {
		return fmt.Errorf("projector ggufPath must be a non-empty path without NUL bytes")
	}
	if model.Type != ModelTypeChat || model.ToolCalling {
		return fmt.Errorf("a media projector requires a chat model without tool calling")
	}
	if !p.Images && !p.Audio && !p.Video {
		return fmt.Errorf("a media projector requires an enabled input capability")
	}
	return nil
}

func (g *generator) loadProjector(ctx context.Context) error {
	if err := g.available(ctx); err != nil {
		return err
	}
	operator, ok := g.operator.(MediaGenerationOperator)
	if !ok {
		return fmt.Errorf("operator does not support media generation")
	}
	var err error
	g.projector, err = operator.LoadProjector(ctx, g.loaded, g.model)
	if err != nil {
		return fmt.Errorf("load projector: %w", err)
	}
	if g.projector == 0 {
		return fmt.Errorf("load projector: operator returned a zero handle")
	}
	return nil
}

func (g *generator) generateMedia(ctx context.Context, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	if err := g.available(ctx); err != nil {
		return nil, err
	}
	operator, ok := g.operator.(MediaGenerationOperator)
	if !ok {
		return nil, fmt.Errorf("yzma: operator does not support media generation")
	}
	constraint, err := prepareConstraint(input)
	if err != nil {
		return nil, err
	}
	var constrainedOperator ConstrainedMediaGenerationOperator
	messageInput := input
	if constraint != nil {
		if !g.model.Projector.Images {
			return nil, fmt.Errorf("yzma: constrained media output requires an image-capable projector")
		}
		if cb != nil {
			return nil, fmt.Errorf("yzma: constrained output does not support streaming")
		}
		constrainedOperator, ok = g.operator.(ConstrainedMediaGenerationOperator)
		if !ok {
			return nil, fmt.Errorf("yzma: the model operator does not support constrained media output")
		}
		// The output config has been checked above; parse only the messages below.
		validated := *input
		validated.Output = nil
		messageInput = &validated
	}
	messages, err := mediaMessages(ctx, messageInput, *g.model.Projector, constraint != nil)
	if err != nil {
		return nil, err
	}
	requestConfig, err := decodeGenerationConfig(input.Config)
	if err != nil {
		return nil, err
	}
	cfg, err := resolveGenerationConfig(nil, g.model.GenerationDefaults, requestConfig)
	if err != nil {
		return nil, err
	}
	if constraint != nil && len(cfg.StopSequences) > 0 {
		return nil, fmt.Errorf("yzma: constrained output does not support stop sequences")
	}
	if err := g.acquire(ctx); err != nil {
		return nil, err
	}
	defer g.release()
	var onText func(context.Context, string) error
	if cb != nil {
		onText = func(ctx context.Context, text string) error {
			return cb(ctx, &ai.ModelResponseChunk{Content: []*ai.Part{ai.NewTextPart(text)}})
		}
	}
	settings := TextGenerationInput{
		Type: ModelTypeChat, ChatTemplate: g.model.ChatTemplate,
		ChatTemplateName: g.model.ChatTemplateName, Context: g.model.Context,
		Sampler: cfg.Sampler, MaxOutputTokens: cfg.MaxOutputTokens, StopSequences: cfg.StopSequences,
	}
	generation := MediaGenerationInput{TextGenerationInput: settings, Messages: messages}
	var result TextGenerationResult
	if constraint != nil {
		result, err = constrainedOperator.GenerateConstrainedMedia(ctx, g.loaded, g.projector, generation, constraint.grammar)
	} else {
		result, err = operator.GenerateMedia(ctx, g.loaded, g.projector, generation, onText)
	}
	if err != nil {
		return nil, fmt.Errorf("generate with model %q: %w", g.model.Name, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if constraint != nil {
		result.Text, err = constraint.outputText(result)
		if err != nil {
			return nil, err
		}
	}
	return generationResponse(input, result, []*ai.Part{ai.NewTextPart(result.Text)}), nil
}

func mediaMessages(ctx context.Context, input *ai.ModelRequest, projector ProjectorOptions, imagesOnly bool) ([]MediaMessage, error) {
	if input == nil || len(input.Messages) == 0 {
		return nil, fmt.Errorf("yzma: at least one message is required")
	}
	if len(input.Tools) > 0 || (input.ToolChoice != "" && input.ToolChoice != ai.ToolChoiceNone && input.ToolChoice != ai.ToolChoiceAuto) {
		return nil, fmt.Errorf("yzma: media models do not support tools")
	}
	if len(input.Docs) > 0 {
		return nil, fmt.Errorf("yzma: native document grounding is not supported")
	}
	if out := input.Output; out != nil && (out.Constrained || len(out.Schema) > 0 ||
		(out.Format != "" && out.Format != ai.OutputFormatText) ||
		(out.ContentType != "" && out.ContentType != "text/plain")) {
		return nil, fmt.Errorf("yzma: media models support only unconstrained text output")
	}
	messages := make([]MediaMessage, 0, len(input.Messages))
	count, budget := 0, maxRequestMedia
	for i, msg := range input.Messages {
		if msg == nil || len(msg.Content) == 0 {
			return nil, fmt.Errorf("yzma: message %d has no content", i)
		}
		if msg.Role != ai.RoleUser && msg.Role != ai.RoleSystem && msg.Role != ai.RoleModel {
			return nil, fmt.Errorf("yzma: message %d has unsupported role %q", i, msg.Role)
		}
		m := MediaMessage{Role: roleMapping[msg.Role]}
		for j, part := range msg.Content {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			switch {
			case part.IsText():
				if strings.ContainsRune(part.Text, '\x00') || strings.Contains(part.Text, mediaSlotPrefix) {
					return nil, fmt.Errorf("yzma: message %d part %d contains a NUL byte or a reserved media marker", i, j)
				}
				if part.Text != "" {
					m.Parts = append(m.Parts, MediaInputPart{Text: part.Text})
				}
			case part.IsMedia() && msg.Role == ai.RoleUser:
				count++
				if count > maxMediaParts {
					return nil, fmt.Errorf("yzma: at most %d media parts per request are supported", maxMediaParts)
				}
				p, err := decodeMediaPart(ctx, part, projector, budget, imagesOnly)
				if err != nil {
					return nil, fmt.Errorf("yzma: message %d part %d: %w", i, j, err)
				}
				budget -= len(p.Data)
				m.Parts = append(m.Parts, p)
			default:
				return nil, fmt.Errorf("yzma: message %d part %d must be text or inline user media", i, j)
			}
		}
		if len(m.Parts) == 0 {
			return nil, fmt.Errorf("yzma: message %d has no content", i)
		}
		messages = append(messages, m)
	}
	return messages, nil
}

func decodeMediaPart(ctx context.Context, part *ai.Part, projector ProjectorOptions, budget int, imagesOnly bool) (MediaInputPart, error) {
	header, payload, hasData := strings.Cut(part.Text, ",")
	mime, mediaURL := strings.CutPrefix(header, "data:")
	mime, isBase64 := strings.CutSuffix(mime, ";base64")
	if !hasData || !mediaURL || !isBase64 {
		return MediaInputPart{}, fmt.Errorf("media must be an inline base64 data URL")
	}
	if part.ContentType != "" && part.ContentType != mime {
		return MediaInputPart{}, fmt.Errorf("media content type does not match the data URL")
	}
	if imagesOnly && !strings.HasPrefix(mime, "image/") {
		return MediaInputPart{}, fmt.Errorf("constrained media output supports only image input")
	}
	switch mime {
	case "image/jpeg", "image/png":
		if projector.Images {
			return decodeImagePart(mime, payload, budget)
		}
	case "audio/wav", "audio/x-wav":
		if projector.Audio {
			return decodeAudioPart(mime, payload, budget)
		}
	case videoMIME:
		if projector.Video {
			return decodeVideoPart(ctx, payload, budget)
		}
	}
	return MediaInputPart{}, fmt.Errorf("unsupported media MIME type %q", mime)
}

func decodeInlineBytes(payload string, limit int) ([]byte, error) {
	if len(payload) > base64.StdEncoding.EncodedLen(limit) {
		return nil, fmt.Errorf("media exceeds the %d-byte limit", limit)
	}
	if payload == "" || strings.ContainsAny(payload, "\r\n") {
		return nil, fmt.Errorf("media must contain non-empty base64 without line breaks")
	}
	b, err := base64.StdEncoding.Strict().DecodeString(payload)
	if err != nil {
		return nil, fmt.Errorf("invalid media base64: %w", err)
	}
	if len(b) > limit {
		return nil, fmt.Errorf("media exceeds the %d-byte limit", limit)
	}
	return b, nil
}
