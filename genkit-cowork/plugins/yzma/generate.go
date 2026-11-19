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
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"unicode/utf8"

	"github.com/firebase/genkit/go/ai"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/message"
)

// TextGenerationOperator optionally extends ModelOperator with text inference.
// Each call must own its inference context and sampler, release both before
// returning, and preserve the supplied model's weights. The plugin serializes
// calls for one registration; different registrations may run concurrently.
// If onText is non-nil, emit valid UTF-8 text without stop sequences and propagate
// callback errors. The final result must still contain the entire response.
// The Genkit adapter buffers tool-enabled output and parses its raw tool syntax.
type TextGenerationOperator interface {
	GenerateText(ctx context.Context, model llama.Model, input TextGenerationInput, onText func(context.Context, string) error) (TextGenerationResult, error)
}

// ConstrainedTextGenerationOperator extends text generation with GBNF decoding.
// Apply grammar starting at the "root" rule to every sampled output token,
// never to the prompt tokens. Report token or context exhaustion as Limited.
// As with GenerateText, the call owns its inference resources.
type ConstrainedTextGenerationOperator interface {
	TextGenerationOperator
	GenerateConstrainedText(ctx context.Context, model llama.Model, input TextGenerationInput, grammar string) (TextGenerationResult, error)
}

// TextGenerationInput contains validated messages and resolved settings.
// Chat messages retain roles, typed calls, IDs, and tool results through Yzma's
// Message interface. Completion accepts exactly one user message.
// Type selects formatting; Sampler is request-local.
type TextGenerationInput struct {
	Type               ModelType
	Messages           []message.Message
	ChatTemplate       string
	ChatTemplateName   string
	ToolCalling        bool
	Tools              []message.ToolDefinition
	Context            ContextOptions
	Sampler            llama.SamplerParams
	MaxOutputTokens    int
	StopSequences      []string
	bosToken, eosToken string
}

// TextGenerationResult contains raw model text and native token counts.
// For tool-enabled models, the Genkit adapter parses and removes tool syntax.
// OutputTokens includes sampled end-of-generation tokens and stripped stop text.
// Limited reports exhaustion of the output limit or remaining context capacity.
type TextGenerationResult struct {
	Text         string
	InputTokens  int
	OutputTokens int
	Limited      bool
}

// textMessages rejects unsupported input instead of dropping content.
func textMessages(input *ai.ModelRequest) ([]message.Chat, error) {
	if input == nil || len(input.Messages) == 0 {
		return nil, fmt.Errorf("yzma: at least one message is required")
	}
	if len(input.Tools) > 0 || (input.ToolChoice != "" && input.ToolChoice != ai.ToolChoiceNone && input.ToolChoice != ai.ToolChoiceAuto) {
		return nil, fmt.Errorf("yzma: tool calling requires the tool-aware chat adapter")
	}
	if len(input.Docs) > 0 {
		return nil, fmt.Errorf("yzma: native document grounding is not supported")
	}
	if out := input.Output; out != nil {
		if out.Constrained || len(out.Schema) > 0 ||
			(out.Format != "" && out.Format != ai.OutputFormatText) ||
			(out.ContentType != "" && out.ContentType != "text/plain") {
			return nil, fmt.Errorf("yzma: only unconstrained text output is supported")
		}
	}
	messages := make([]message.Chat, 0, len(input.Messages))
	for i, msg := range input.Messages {
		if msg == nil {
			return nil, fmt.Errorf("yzma: message %d is nil", i)
		}
		switch msg.Role {
		case ai.RoleSystem, ai.RoleUser, ai.RoleModel:
		default:
			return nil, fmt.Errorf("yzma: message %d has unsupported role %q", i, msg.Role)
		}
		var content strings.Builder
		for _, part := range msg.Content {
			if !part.IsText() {
				return nil, fmt.Errorf("yzma: message %d contains non-text content", i)
			}
			content.WriteString(part.Text)
		}
		text := content.String()
		if text == "" || strings.ContainsRune(text, '\x00') {
			return nil, fmt.Errorf("yzma: message %d must contain non-empty text without NUL bytes", i)
		}
		messages = append(messages, message.Chat{Role: roleMapping[msg.Role], Content: text})
	}
	return messages, nil
}

// decodeGenerationConfig accepts Genkit's typed and JSON-decoded configuration.
// any is required here because ModelRequest.Config is defined by Genkit as any.
func decodeGenerationConfig(value any) (*GenerateContentConfig, error) {
	switch cfg := value.(type) {
	case nil:
		return nil, nil
	case GenerateContentConfig:
		return &cfg, nil
	case *GenerateContentConfig:
		return cfg, nil
	case map[string]any:
		data, err := json.Marshal(cfg)
		if err != nil {
			return nil, fmt.Errorf("yzma: encode generation config: %w", err)
		}
		var result GenerateContentConfig
		decoder := json.NewDecoder(bytes.NewReader(data))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&result); err != nil {
			return nil, fmt.Errorf("yzma: decode generation config: %w", err)
		}
		return &result, nil
	default:
		return nil, fmt.Errorf("yzma: unsupported config type %T; use GenerateContentConfig", value)
	}
}

func (defaultModelOperator) GenerateText(ctx context.Context, model llama.Model, input TextGenerationInput, onText func(context.Context, string) error) (result TextGenerationResult, err error) {
	return generateNativeText(ctx, model, input, "", onText)
}

func (defaultModelOperator) GenerateConstrainedText(ctx context.Context, model llama.Model, input TextGenerationInput, grammar string) (TextGenerationResult, error) {
	if grammar == "" {
		return TextGenerationResult{}, fmt.Errorf("missing output grammar")
	}
	return generateNativeText(ctx, model, input, grammar, nil)
}

func generateNativeText(ctx context.Context, model llama.Model, input TextGenerationInput, grammar string, onText func(context.Context, string) error) (result TextGenerationResult, err error) {
	if err := ctx.Err(); err != nil {
		return result, err
	}
	if llama.ModelHasEncoder(model) || !llama.ModelHasDecoder(model) {
		return result, fmt.Errorf("only decoder-only text models are supported")
	}
	vocab := llama.ModelGetVocab(model)
	input.bosToken = tokenSpelling(vocab, llama.VocabBOS(vocab))
	input.eosToken = tokenSpelling(vocab, llama.VocabEOS(vocab))
	prompt, parseSpecial, err := prepareTextPrompt(input,
		func() string { return embeddedChatTemplate(model, input.ChatTemplateName, len(input.Tools) > 0) }, llama.ChatApplyTemplate)
	if err != nil {
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
	addSpecial := !parseSpecial || input.bosToken == "" || !strings.HasPrefix(prompt, input.bosToken)
	tokens := llama.Tokenize(vocab, prompt, addSpecial, parseSpecial)
	capacity, batchSize := int(llama.NCtxSeq(lctx)), int(llama.NBatch(lctx))
	if len(tokens) == 0 || len(tokens) >= capacity {
		return result, fmt.Errorf("prompt has %d tokens; context capacity is %d and must leave room for output", len(tokens), capacity)
	}
	sampler := textSampler(model, input.Sampler)
	if sampler == 0 {
		return result, fmt.Errorf("allocate sampler")
	}
	defer llama.SamplerFree(sampler)
	for _, token := range tokens {
		llama.SamplerAccept(sampler, token)
	}
	if grammar != "" {
		if err := addOutputGrammar(sampler, vocab, grammar); err != nil {
			return result, err
		}
	}
	if input.ToolCalling {
		input.StopSequences = append(append([]string(nil), input.StopSequences...),
			message.StopMarkers(vocab, message.FormatAuto)...)
		input.StopSequences = append(input.StopSequences, "<|tool_response>", "<|tool_response_start|>")
	}
	session := nativeTokenSession{ctx: lctx, vocab: vocab, sampler: sampler, special: input.ToolCalling}
	return generateTokens(ctx, session, tokens, batchSize, capacity, input, onText)
}

// addOutputGrammar runs after the sampler has accepted the prompt. Move the
// existing stages behind the grammar so it sees the full vocabulary.
func addOutputGrammar(sampler llama.Sampler, vocab llama.Vocab, grammar string) error {
	filter := llama.SamplerInitGrammar(vocab, grammar, "root")
	if filter == 0 {
		return fmt.Errorf("allocate output grammar sampler")
	}
	stages := llama.SamplerChainN(sampler)
	llama.SamplerChainAdd(sampler, filter)
	for range stages {
		llama.SamplerChainAdd(sampler, llama.SamplerChainRemove(sampler, 0))
	}
	return nil
}

// prepareTextPrompt keeps completion text literal. Only chat mode reads model
// template metadata or interprets template-produced special-token spellings.
func prepareTextPrompt(input TextGenerationInput, modelTemplate func() string, apply func(string, []llama.ChatMessage, bool, []byte) int32) (string, bool, error) {
	switch input.Type {
	case ModelTypeCompletion:
		if input.ChatTemplate != "" || input.ChatTemplateName != "" || input.ToolCalling || len(input.Tools) > 0 {
			return "", false, fmt.Errorf("completion models must not specify a chat template")
		}
		if len(input.Messages) != 1 || input.Messages[0].GetRole() != "user" {
			return "", false, fmt.Errorf("completion models require exactly one user text message")
		}
		content, ok := input.Messages[0].GetContent()["content"].(string)
		if !ok {
			return "", false, fmt.Errorf("completion requires a text prompt")
		}
		return content, false, nil
	case ModelTypeChat:
		template := input.ChatTemplate
		if template == "" {
			template = modelTemplate()
		}
		if template == "" && input.ChatTemplateName != "" {
			return "", false, fmt.Errorf("embedded chat template %q was not found", input.ChatTemplateName)
		}
		if template == "" || strings.ContainsRune(template, '\x00') {
			return "", false, fmt.Errorf("no valid chat template; set ModelDefinition.ChatTemplate explicitly")
		}
		if source, ok := builtinChatTemplate(template); ok {
			template = source
		}
		if strings.Contains(template, "{{") || strings.Contains(template, "{%") {
			prompt, err := renderJinjaPrompt(template, input)
			return prompt, true, err
		}
		if input.ToolCalling {
			return "", false, fmt.Errorf("tool calling requires a Jinja template that formats tools and tool history")
		}
		chat := make([]llama.ChatMessage, len(input.Messages))
		for i, msg := range input.Messages {
			content, ok := msg.GetContent()["content"].(string)
			if !ok {
				return "", false, fmt.Errorf("native chat formatting requires text content")
			}
			chat[i] = llama.NewChatMessage(msg.GetRole(), content)
		}
		prompt, err := renderChatPrompt(template, chat, apply)
		runtime.KeepAlive(chat)
		return prompt, true, err
	default:
		return "", false, fmt.Errorf("unsupported model type %q", input.Type)
	}
}

// renderChatPrompt uses llama.cpp's template support without silently substituting
// another model's format. The apply function also permits testing buffer resizing.
func renderChatPrompt(template string, chat []llama.ChatMessage, apply func(string, []llama.ChatMessage, bool, []byte) int32) (string, error) {
	buf := make([]byte, 4096)
	n := apply(template, chat, true, buf)
	if int(n) > len(buf) {
		buf = make([]byte, n)
		n = apply(template, chat, true, buf)
	}
	if n <= 0 || int(n) > len(buf) {
		return "", fmt.Errorf("chat template is unsupported or invalid; set a supported ModelDefinition.ChatTemplate")
	}
	return string(buf[:n]), nil
}

func textSampler(model llama.Model, params llama.SamplerParams) llama.Sampler {
	if params.Temp == 0 {
		chain := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
		if chain == 0 {
			return 0
		}
		llama.SamplerChainAdd(chain, llama.SamplerInitPenalties(
			llama.VocabNTokens(llama.ModelGetVocab(model)), params.PenaltyLastN,
			params.PenaltyRepeat, params.PenaltyFreq, params.PenaltyPresent,
		))
		llama.SamplerChainAdd(chain, llama.SamplerInitGreedy())
		return chain
	}
	return llama.NewSampler(model, []llama.SamplerType{
		llama.SamplerTypePenalties, llama.SamplerTypeTopK, llama.SamplerTypeTopP,
		llama.SamplerTypeMinP, llama.SamplerTypeTemperature,
	}, &params)
}

// tokenSession is the decode-loop boundary. Native allocation is owned by the
// caller, while tests can supply a deterministic token sequence.
type tokenSession interface {
	Decode(ctx context.Context, tokens []llama.Token) error
	Sample(ctx context.Context) (llama.Token, error)
	IsEnd(token llama.Token) bool
	Piece(token llama.Token) (string, error)
}

type nativeTokenSession struct {
	ctx     llama.Context
	vocab   llama.Vocab
	sampler llama.Sampler
	special bool
	// Media prefill can advance positions differently from the token count.
	position *llama.Pos
}

func (s nativeTokenSession) Decode(ctx context.Context, tokens []llama.Token) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	// Text prefill uses automatic positions for batches. Media prefill is
	// handled by mtmd; only its single-token sampling steps set an explicit
	// position. llama_decode reads one position per token when Pos is nonnil.
	if s.position != nil && len(tokens) != 1 {
		return fmt.Errorf("decode at an explicit position requires exactly one token")
	}
	batch := llama.BatchGetOne(tokens)
	batch.Pos = s.position
	code, err := llama.Decode(s.ctx, batch)
	runtime.KeepAlive(tokens)
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if err != nil {
		return err
	}
	if code != 0 {
		return fmt.Errorf("llama.Decode returned status %d", code)
	}
	if s.position != nil {
		*s.position++
	}
	return nil
}

func (s nativeTokenSession) Sample(ctx context.Context) (llama.Token, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	token := llama.SamplerSample(s.sampler, s.ctx, -1)
	if token == llama.TokenNull {
		return 0, fmt.Errorf("sampler returned an invalid token")
	}
	return token, nil
}

func (s nativeTokenSession) IsEnd(token llama.Token) bool {
	return llama.VocabIsEOG(s.vocab, token)
}

func (s nativeTokenSession) Piece(token llama.Token) (string, error) {
	buf := make([]byte, 256)
	n := llama.TokenToPiece(s.vocab, token, buf, 0, s.special)
	if n < 0 {
		buf = make([]byte, -int64(n))
		n = llama.TokenToPiece(s.vocab, token, buf, 0, s.special)
	}
	if n < 0 || int(n) > len(buf) {
		return "", fmt.Errorf("unable to decode token %d", token)
	}
	return string(buf[:n]), nil
}

// generateTokens evaluates prompts in logical batches, then decodes one token at
// a time. It counts actual samples rather than prompt batches toward the limit.
func generateTokens(ctx context.Context, session tokenSession, prompt []llama.Token, batchSize, capacity int, input TextGenerationInput, onText func(context.Context, string) error) (TextGenerationResult, error) {
	result := TextGenerationResult{InputTokens: len(prompt)}
	if len(prompt) == 0 || len(prompt) >= capacity || batchSize <= 0 || input.MaxOutputTokens <= 0 {
		return result, fmt.Errorf("invalid prompt capacity, batch size, or output limit")
	}
	for start := 0; start < len(prompt); start += batchSize {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if err := session.Decode(ctx, prompt[start:min(start+batchSize, len(prompt))]); err != nil {
			return result, fmt.Errorf("decode prompt: %w", err)
		}
	}
	return sampleTokens(ctx, session, len(prompt), capacity-len(prompt), input, onText)
}

// sampleTokens starts from logits left by a successful text or media prefill.
func sampleTokens(ctx context.Context, session tokenSession, inputTokens, remaining int, input TextGenerationInput, onText func(context.Context, string) error) (TextGenerationResult, error) {
	result := TextGenerationResult{InputTokens: inputTokens}
	output := textOutput{stops: input.StopSequences, onText: onText}
	limit := min(input.MaxOutputTokens, remaining)
	for range limit {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		token, err := session.Sample(ctx)
		if err != nil {
			return result, err
		}
		result.OutputTokens++
		if session.IsEnd(token) {
			_, err := output.flush(ctx, true)
			result.Text = output.text.String()
			return result, err
		}
		piece, err := session.Piece(token)
		if err != nil {
			return result, err
		}
		output.pending += piece
		stopped, err := output.flush(ctx, false)
		if err != nil {
			return result, err
		}
		if stopped {
			result.Text = output.text.String()
			return result, nil
		}
		if result.OutputTokens < limit {
			if err := session.Decode(ctx, []llama.Token{token}); err != nil {
				return result, fmt.Errorf("decode output: %w", err)
			}
		}
	}
	_, err := output.flush(ctx, true)
	result.Text, result.Limited = output.text.String(), true
	return result, err
}

// textOutput retains incomplete UTF-8 and potential stop-sequence prefixes.
// Only text that cannot become a stop sequence is sent to the callback.
type textOutput struct {
	text    strings.Builder
	pending string
	stops   []string
	onText  func(context.Context, string) error
}

func (o *textOutput) flush(ctx context.Context, final bool) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	end, stopped := len(o.pending), false
	for _, stop := range o.stops {
		if i := strings.Index(o.pending, stop); i >= 0 && i < end {
			end, stopped = i, true
		}
	}
	if !final && !stopped {
		for _, stop := range o.stops {
			for n := min(len(stop)-1, len(o.pending)); n > 0; n-- {
				if strings.HasSuffix(o.pending, stop[:n]) {
					end = min(end, len(o.pending)-n)
					break
				}
			}
		}
		for pos := 0; pos < end; {
			if !utf8.FullRuneInString(o.pending[pos:end]) {
				end = pos
				break
			}
			_, n := utf8.DecodeRuneInString(o.pending[pos:end])
			pos += n
		}
	}
	text := strings.ToValidUTF8(o.pending[:end], "\uFFFD")
	o.pending = o.pending[end:]
	if stopped {
		o.pending = ""
	}
	if text != "" {
		if o.onText != nil {
			if err := o.onText(ctx, text); err != nil {
				return stopped, err
			}
		}
		o.text.WriteString(text)
	}
	return stopped, nil
}
