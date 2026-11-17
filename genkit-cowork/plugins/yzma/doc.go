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
// explicit preloading, and cleanup. Token generation is not implemented yet.
//
// Pass a Yzma instance to genkit.WithPlugins, then call DefineModel for each
// model. Preload checks loading before the application starts serving requests.
// Close rejects new work and frees weights after active use completes.
// Native shared libraries and backends remain initialized for the process
// lifetime; all default-operator plugins must use the same library directory.
// Applications must not independently unload the Yzma runtime while using this
// plugin.
//
// Native loading requires llama.cpp libraries compatible with the pinned Yzma
// version. See https://yzma.ai/getting-started/install/ for installation.
// The examples/yzma-loading program demonstrates loading and releasing a model
// without allocating an inference context or generating tokens.
package yzma
