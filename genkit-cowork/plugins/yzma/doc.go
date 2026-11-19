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

// Package yzma registers local GGUF models with Genkit using in-process llama.cpp
// bindings. It supports metadata-only model discovery, lazy weight loading,
// explicit preloading, streaming text generation, and cleanup. Generation supports
// decoder-only models, chat or raw completion, sampling settings, and stop sequences.
// Chat models can opt into tool calling through ModelDefinition.ToolCalling.
// Genkit executes the tools; the plugin renders schemas and history and parses
// calls using Yzma's family parsers. Chat and completion can generate constrained
// JSON objects through genkit.GenerateData, without tools or streaming. See the
// package README for the supported schema subset. Vision chat accepts ordered
// inline JPEG and PNG parts through a compatible projector. It supports
// constrained JSON without tools. GenerateBoundingBox returns a validated,
// typed box for an image and instruction and does not execute actions.
// Qwen3-ASR chat accepts ordered inline PCM WAV parts through a compatible
// projector, without tools or structured output.
// ModelTypeSpeech turns one text prompt into a WAV clip
// using a Soprano backbone and matching decoder. It does not accept media input.
// ModelTypeDecision uses Yzma's local System One readouts with typed Choice,
// Score, and Noul fields.
//
// Pass a Yzma instance to genkit.WithPlugins to discover models in ModelsDir
// or Yzma's default directory. The catalog is fixed at initialization. Restart
// to pick up file changes. Use DefineModel for overrides or unclassified files.
// Preload checks loading before the application starts serving requests.
// Manual definitions require ModelDefinition.Type. ModelTypeChat uses a chat
// template, while ModelTypeCompletion accepts one user prompt without formatting.
// Embedded GGUF templates are rendered with a full Jinja engine, including
// named templates, BOS/EOS variables, and typed tool arguments and results.
// Each plugin instance loads at most one model at a time. Close rejects new
// work and frees projectors, speech decoders, and model weights after active use.
// A failed companion load frees the companion before its model. If cleanup
// fails, the next load attempt or Close retries it before proceeding.
// Native shared libraries and backends remain initialized for the process
// lifetime; all default-operator plugins must use the same library directory.
// Applications must not independently unload the Yzma runtime while using this
// plugin.
//
// Local text embedders use genkit.Embed with discovered BERT GGUFs or
// DefineEmbedder for explicit configuration. Each document produces one
// L2-normalized vector with the model's fixed output dimensions.
//
// Native loading requires llama.cpp libraries compatible with the pinned Yzma
// version. See https://yzma.ai/getting-started/install/ for installation.
// The examples/yzma-loading program demonstrates discovery in the Genkit Dev UI,
// explicit registration, and cleanup. See examples/yzma-structured for
// constrained generation, examples/yzma-vision for inline images and bounding boxes,
// examples/yzma-speech for saving a WAV, examples/yzma-audio for transcription,
// and examples/yzma-decision for local System One decisions.
package yzma
