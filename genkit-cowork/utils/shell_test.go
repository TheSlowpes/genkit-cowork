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

package utils

import (
	"strings"
	"testing"
)

func TestSplitPathList(t *testing.T) {
	got := splitPathList("a::b:", ":")
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("splitPathList() = %#v, want [\"a\",\"b\"]", got)
	}
}

func TestMapToEnvSlice(t *testing.T) {
	out := mapToEnvSlice(map[string]string{"A": "1"})
	if len(out) != 1 || !strings.HasPrefix(out[0], "A=") {
		t.Fatalf("mapToEnvSlice() = %#v, want one A= entry", out)
	}
}

func TestGetShellEnv(t *testing.T) {
	got := GetShellEnv()
	if len(got) == 0 {
		t.Fatal("GetShellEnv() returned empty env")
	}
}
