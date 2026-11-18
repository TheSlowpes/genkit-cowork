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
// bindings. It currently supports configuration validation, lazy weight loading,
// explicit preloading, streaming text generation, and cleanup. Generation supports
// decoder-only models, chat or raw completion, sampling settings, and stop sequences.
// Chat models can opt into tool calling through ModelDefinition.ToolCalling.
// Genkit executes the tools; the plugin renders schemas and history and parses
// calls using Yzma's family parsers. Chat and completion can generate constrained
// JSON objects through genkit.GenerateData, without tools or streaming. See the
// package README for the supported schema subset. Media input is unsupported.
//
// Pass a Yzma instance to genkit.WithPlugins, then call DefineModel for each
// model. Preload checks loading before the application starts serving requests.
// ModelDefinition.Type is required: ModelTypeChat uses a chat template, while
// ModelTypeCompletion accepts a single user text prompt without chat formatting.
// Embedded GGUF templates are rendered with a full Jinja engine, including
// named templates, BOS/EOS variables, and typed tool arguments and results.
// Close rejects new work and frees weights after active use completes.
// Native shared libraries and backends remain initialized for the process
// lifetime; all default-operator plugins must use the same library directory.
// Applications must not independently unload the Yzma runtime while using this
// plugin.
//
// Native loading requires llama.cpp libraries compatible with the pinned Yzma
// version. See https://yzma.ai/getting-started/install/ for installation.
// The examples/yzma-loading program demonstrates loading and releasing a model
// and optionally generating text. Its -serve flag keeps the process running for
// the Genkit Dev UI. See examples/yzma-structured for typed output.
package yzma
