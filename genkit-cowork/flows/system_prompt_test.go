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

package flows

import (
	"context"
	"strings"
	"testing"
	"time"
)

// --- Helpers ---

// promptFromOpts builds the prompt string directly (bypasses PromptFn wrapper).
func promptFromOpts(opts SystemPromptOptions) string {
	return buildPromptString(opts)
}

// invokePromptFn calls the PromptFn and returns the resulting string.
func invokePromptFn(t *testing.T, opts SystemPromptOptions) string {
	t.Helper()
	fn := BuildSystemPrompt(opts)
	text, err := fn(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildSystemPrompt returned unexpected error: %v", err)
	}
	return text
}

// --- Tests: DefaultSystemPrompt ---

func TestDefaultSystemPrompt_ContainsExpectedSections(t *testing.T) {
	fn := DefaultSystemPrompt()
	text, err := fn(context.Background(), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	checks := []string{
		"capable assistant",
		"get work done",
		"Guidelines:",
		"Current date:",
		"Current working directory:",
	}
	for _, want := range checks {
		if !strings.Contains(text, want) {
			t.Errorf("expected prompt to contain %q\nfull prompt:\n%s", want, text)
		}
	}

	// Tool list and skills are sent by Genkit natively — they must not appear
	// as redundant text sections in the system prompt.
	absent := []string{"Available tools:", "# Skills"}
	for _, unwanted := range absent {
		if strings.Contains(text, unwanted) {
			t.Errorf("prompt must not contain %q (Genkit sends this natively)\nfull prompt:\n%s", unwanted, text)
		}
	}
}

func TestDefaultSystemPrompt_DateIsToday(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{})
	today := time.Now().Format("2006-01-02")
	if !strings.Contains(text, "Current date: "+today) {
		t.Errorf("expected 'Current date: %s' in prompt\nfull prompt:\n%s", today, text)
	}
}

func TestDefaultSystemPrompt_CwdIsPresent(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{})
	if !strings.Contains(text, "Current working directory:") {
		t.Errorf("expected 'Current working directory:' in prompt\nfull prompt:\n%s", text)
	}
}

// --- Tests: SelectedTools drives guidelines ---

func TestBuildSystemPrompt_SelectedToolsDriveGuidelines(t *testing.T) {
	// When only bash+read are selected, the edit/write guidelines must be absent.
	text := invokePromptFn(t, SystemPromptOptions{
		SelectedTools: []string{"bash", "read"},
	})

	if !strings.Contains(text, "Use bash for file operations like ls, find, grep") {
		t.Error("expected bash guideline when bash is selected")
	}
	if strings.Contains(text, "Use edit for surgical") {
		t.Error("did not expect edit guideline when edit is not selected")
	}
	if strings.Contains(text, "Use write only for new files") {
		t.Error("did not expect write guideline when write is not selected")
	}
}

// --- Tests: Guidelines auto-generation ---

func TestBuildSystemPrompt_GuidelinesIncludeBashWhenBashSelected(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		SelectedTools: []string{"bash"},
	})
	if !strings.Contains(text, "Use bash for file operations like ls, find, grep") {
		t.Error("expected bash guideline when bash tool is selected")
	}
}

func TestBuildSystemPrompt_GuidelinesIncludeReadBeforeEditWhenBothSelected(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		SelectedTools: []string{"read", "edit"},
	})
	if !strings.Contains(text, "Use read to examine files before editing") {
		t.Error("expected read-before-edit guideline")
	}
}

func TestBuildSystemPrompt_ReadBeforeEditGuidelineAbsentWithoutEdit(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		SelectedTools: []string{"read"},
	})
	if strings.Contains(text, "Use read to examine files before editing") {
		t.Error("did not expect read-before-edit guideline when edit tool is absent")
	}
}

func TestBuildSystemPrompt_EditGuidelinePresent(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		SelectedTools: []string{"edit"},
	})
	if !strings.Contains(text, "Use edit for surgical, targeted changes to existing files") {
		t.Error("expected edit guideline when edit tool is selected")
	}
}

func TestBuildSystemPrompt_WriteGuidelinePresent(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		SelectedTools: []string{"write"},
	})
	if !strings.Contains(text, "Use write only for new files or complete rewrites") {
		t.Error("expected write guideline when write tool is selected")
	}
}

func TestBuildSystemPrompt_PlainTextOutputGuidelineForEditOrWrite(t *testing.T) {
	for _, tool := range []string{"edit", "write"} {
		text := invokePromptFn(t, SystemPromptOptions{
			SelectedTools: []string{tool},
		})
		if !strings.Contains(text, "output plain text directly") {
			t.Errorf("expected plain-text output guideline when %s tool is selected", tool)
		}
	}
}

func TestBuildSystemPrompt_AlwaysPresentGuidelines(t *testing.T) {
	// Even with no tools, the universal guidelines must appear.
	text := invokePromptFn(t, SystemPromptOptions{
		SelectedTools: []string{},
	})
	for _, want := range []string{
		"Be concise in your responses",
		"Show file paths clearly when working with files",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("expected universal guideline %q to always appear\nfull prompt:\n%s", want, text)
		}
	}
}

func TestBuildSystemPrompt_ExtraGuidelinesAreAppended(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		Guidelines: []string{"Always write tests", "Prefer interfaces over concrete types"},
	})
	for _, want := range []string{
		"Always write tests",
		"Prefer interfaces over concrete types",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("expected extra guideline %q in prompt", want)
		}
	}
}

func TestBuildSystemPrompt_DuplicateGuidelinesDropped(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		SelectedTools: []string{"bash"},
		// Duplicate the auto-generated bash guideline.
		Guidelines: []string{"Use bash for file operations like ls, find, grep"},
	})
	count := strings.Count(text, "Use bash for file operations like ls, find, grep")
	if count != 1 {
		t.Errorf("expected duplicate guideline to appear exactly once, got %d occurrences", count)
	}
}

// --- Tests: CustomPrompt ---

func TestBuildSystemPrompt_CustomPromptReplacesDefaultBody(t *testing.T) {
	const custom = "You are a specialized data analyst."
	text := invokePromptFn(t, SystemPromptOptions{
		CustomPrompt: custom,
	})
	if !strings.Contains(text, custom) {
		t.Error("expected custom prompt to appear in output")
	}
	if strings.Contains(text, "Guidelines:") {
		t.Error("did not expect 'Guidelines:' section when CustomPrompt is set")
	}
}

func TestBuildSystemPrompt_CustomPromptStillGetsCwdAndDate(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		CustomPrompt: "Custom body.",
		Cwd:          "/custom/cwd",
	})
	if !strings.Contains(text, "Current working directory: /custom/cwd") {
		t.Error("expected custom cwd in footer with custom prompt")
	}
	if !strings.Contains(text, "Current date:") {
		t.Error("expected date footer with custom prompt")
	}
}

// --- Tests: AppendPrompt ---

func TestBuildSystemPrompt_AppendPromptIsIncluded(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		AppendPrompt: "IMPORTANT: Never delete files.",
	})
	if !strings.Contains(text, "IMPORTANT: Never delete files.") {
		t.Error("expected AppendPrompt content in output")
	}
}

func TestBuildSystemPrompt_AppendPromptWithCustomPrompt(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		CustomPrompt: "Custom.",
		AppendPrompt: "Appended section.",
	})
	if !strings.Contains(text, "Custom.") {
		t.Error("expected custom prompt")
	}
	if !strings.Contains(text, "Appended section.") {
		t.Error("expected appended content with custom prompt")
	}
}

// --- Tests: ContextFiles ---

func TestBuildSystemPrompt_ContextFilesSection(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		ContextFiles: []ContextFile{
			{Path: "CONTRIBUTING.md", Content: "Always run tests before committing."},
			{Path: ".agent/instructions.md", Content: "Use the staging environment."},
		},
	})
	for _, want := range []string{
		"# Project Context",
		"## CONTRIBUTING.md",
		"Always run tests before committing.",
		"## .agent/instructions.md",
		"Use the staging environment.",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("expected %q in context files section\nfull prompt:\n%s", want, text)
		}
	}
}

func TestBuildSystemPrompt_NoContextFilesSection(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{})
	if strings.Contains(text, "# Project Context") {
		t.Error("did not expect Project Context section when no context files provided")
	}
}

// --- Tests: Cwd option ---

func TestBuildSystemPrompt_CwdOption(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		Cwd: "/home/user/projects/myapp",
	})
	if !strings.Contains(text, "Current working directory: /home/user/projects/myapp") {
		t.Errorf("expected custom cwd in footer\nfull prompt:\n%s", text)
	}
}

func TestBuildSystemPrompt_CwdBackslashesNormalized(t *testing.T) {
	text := invokePromptFn(t, SystemPromptOptions{
		Cwd: `C:\Users\user\projects`,
	})
	if !strings.Contains(text, "Current working directory: C:/Users/user/projects") {
		t.Errorf("expected backslashes to be normalised to forward slashes\nfull prompt:\n%s", text)
	}
}

// --- Tests: BuildSystemPrompt returns a callable ai.PromptFn ---

func TestBuildSystemPrompt_ReturnsCallablePromptFn(t *testing.T) {
	fn := BuildSystemPrompt(SystemPromptOptions{
		Cwd: "/tmp/test",
	})
	// Call twice to confirm the function is reusable.
	for i := range 2 {
		text, err := fn(context.Background(), nil)
		if err != nil {
			t.Fatalf("call %d: unexpected error: %v", i, err)
		}
		if !strings.Contains(text, "Current working directory: /tmp/test") {
			t.Errorf("call %d: expected cwd in output", i)
		}
	}
}
