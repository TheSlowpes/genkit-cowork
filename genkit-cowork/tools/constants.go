// Copyright 2025 Kevin Lopes
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

package tools

// DEFAULT_MAX_LINES is the default maximum number of lines returned by tools
// that truncate textual output.
const (
	// DEFAULT_MAX_LINES is the default line cap for tool output.
	DEFAULT_MAX_LINES = 2000
	// DEFAULT_MAX_BYTES is the default byte cap for tool output.
	DEFAULT_MAX_BYTES = 50 * 1024
	// CONTEXT_LINES is the number of surrounding lines included in edit diffs.
	CONTEXT_LINES = 4
)
