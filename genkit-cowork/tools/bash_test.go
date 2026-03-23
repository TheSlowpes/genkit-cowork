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

package tools

import "testing"

func TestProcessBashOutput_NoOutput(t *testing.T) {
	out := processBashOutput("", 0)
	if out.Content != "(no output)" {
		t.Fatalf("processBashOutput().Content = %q, want %q", out.Content, "(no output)")
	}
}

func TestProcessBashOutput_Truncates(t *testing.T) {
	long := ""
	for i := range 2200 {
		if i > 0 {
			long += "\n"
		}
		long += "line"
	}
	out := processBashOutput(long, 0)
	if out.FullOutputPath == "" {
		t.Fatal("processBashOutput().FullOutputPath is empty, want temp file path")
	}
	if out.Content == long {
		t.Fatal("processBashOutput().Content was not truncated")
	}
}

func TestResolveSpawnContext_Hook(t *testing.T) {
	hook := BashSpawnHook(func(in BashSpawnContext) BashSpawnContext {
		in.Cmd = "echo changed"
		return in
	})
	got := resolveSpawnContext("echo test", "/tmp", &hook)
	if got.Cmd != "echo changed" {
		t.Fatalf("resolveSpawnContext().Cmd = %q, want %q", got.Cmd, "echo changed")
	}
}
