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

import (
	"context"
	"fmt"

	"github.com/TheSlowpes/genkit-cowork/genkit-cowork/memory"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

func ExampleNewSearchSessionMemoryTool() {
	ctx := context.Background()
	g := genkit.Init(ctx)
	retriever := &mockSessionRetriever{
		messages: []memory.SessionMessage{{
			MessageID: "m1",
			Kind:      memory.KindEpisodic,
			Origin:    memory.UIMessage,
			Content:   *ai.NewUserTextMessage("invoice for march"),
		}},
	}

	_ = NewSearchSessionMemoryTool(g, retriever)
	fmt.Println(formatMemoryResults(retriever.messages))
	// Output: 1. id=m1 kind=episodic origin=ui text="invoice for march"
}
