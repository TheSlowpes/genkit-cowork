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

// Package memory provides persistence primitives for conversational agent
// memory.
//
// # Overview
//
// The package stores session state as structured data that can be persisted in
// memory, files, or custom backends. Message records include origin, kind, and
// timestamps so callers can preserve a complete interaction timeline while
// still applying load-time pruning policies.
//
// Vector-backed retrieval is optional. A SessionOperator can be wrapped with a
// VectorOperator to index textual message content for semantic recall while
// keeping the canonical session state in the primary operator.
//
// # Examples
//
// Create a default in-memory store:
//
//	store := memory.NewSession()
//
// Create a file-backed store with vector indexing:
//
//	fileOp := memory.NewFileSessionOperator("./data/sessions")
//	vecOp := memory.NewVectorOperator(fileOp, backend, "./data/sessions")
//	store := memory.NewSession(memory.WithCustomSessionOperator(vecOp))
package memory
