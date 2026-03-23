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
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
)

// defaultTools is the canonical ordered tool set used when SelectedTools is
// not specified.
var defaultTools = []string{"bash", "read", "edit", "write"}

// ContextFile is a named file whose contents are embedded in the system prompt
// as project-specific context.
type ContextFile struct {
	// Path is displayed as the section header in the Project Context section.
	Path string
	// Content is the full file content to embed.
	Content string
}

// SystemPromptOptions configures the prompt produced by BuildSystemPrompt.
type SystemPromptOptions struct {
	// CustomPrompt replaces the auto-generated body when non-empty. The
	// guidelines section is omitted, but AppendPrompt, ContextFiles, and the
	// date/cwd footer are still appended.
	CustomPrompt string

	// SelectedTools is the ordered list of tool names active for the agent.
	// Defaults to ["bash", "read", "edit", "write"] when nil or empty.
	// Used only to derive tool-specific usage guidelines; Genkit sends the
	// actual tool definitions to the model separately.
	SelectedTools []string

	// Guidelines are additional bullet points appended after the
	// auto-generated guidelines. Duplicates are silently dropped.
	Guidelines []string

	// AppendPrompt is free-form text appended to the main prompt body before
	// context files and the date/cwd footer.
	AppendPrompt string

	// Cwd is the working directory reported in the prompt footer.
	// Defaults to the process working directory when empty.
	Cwd string

	// ContextFiles are project-specific instruction files embedded under a
	// "Project Context" heading at the end of the prompt.
	ContextFiles []ContextFile
}

// BuildSystemPrompt returns an ai.PromptFn that generates a system prompt for
// a genkit-cowork agent.
//
// The prompt includes:
//   - Usage guidelines derived automatically from the selected tool set,
//     extended by any caller-supplied Guidelines.
//   - An optional appended free-form section (AppendPrompt).
//   - An optional "Project Context" section from ContextFiles.
//   - A footer with the current date and working directory.
//
// Tool definitions and skill listings are intentionally omitted from the
// prompt text: Genkit sends tool schemas to the model natively, and the
// Skills plugin exposes skills through its own registered tools.
//
// The returned function re-evaluates the current date on every call, so
// long-running agents always report the correct date.
func BuildSystemPrompt(opts SystemPromptOptions) ai.PromptFn {
	return func(ctx context.Context, _ any) (string, error) {
		return buildPromptString(opts), nil
	}
}

// DefaultSystemPrompt returns an ai.PromptFn that generates a system prompt
// using the default tool set (bash, read, edit, write), the process working
// directory, and today's date. It is a convenience wrapper around
// BuildSystemPrompt with zero configuration.
func DefaultSystemPrompt() ai.PromptFn {
	return BuildSystemPrompt(SystemPromptOptions{})
}

// buildPromptString is the pure, testable core of the system prompt builder.
func buildPromptString(opts SystemPromptOptions) string {
	cwd := opts.Cwd
	if cwd == "" {
		if wd, err := os.Getwd(); err == nil {
			cwd = wd
		}
	}
	// Normalise path separators for cross-platform consistency.
	cwd = strings.ReplaceAll(cwd, "\\", "/")

	date := time.Now().Format("2006-01-02")

	tools := opts.SelectedTools
	if len(tools) == 0 {
		tools = defaultTools
	}

	toolSet := make(map[string]bool, len(tools))
	for _, t := range tools {
		toolSet[t] = true
	}

	hasBash := toolSet["bash"]
	hasRead := toolSet["read"]
	hasEdit := toolSet["edit"]
	hasWrite := toolSet["write"]

	var appendSection string
	if opts.AppendPrompt != "" {
		appendSection = "\n\n" + opts.AppendPrompt
	}

	// --- Assemble main body ---
	var prompt string
	if opts.CustomPrompt != "" {
		prompt = opts.CustomPrompt
	} else {
		guidelinesText := buildGuidelinesText(hasBash, hasRead, hasEdit, hasWrite, opts.Guidelines)

		prompt = fmt.Sprintf(
			"You are a capable assistant that helps users get work done. "+
				"You can read and write files, run commands, and make targeted changes to existing content.\n\n"+
				"Guidelines:\n%s",
			guidelinesText,
		)
	}

	if appendSection != "" {
		prompt += appendSection
	}

	// --- Context files ---
	if len(opts.ContextFiles) > 0 {
		prompt += "\n\n# Project Context\n\nProject-specific instructions and guidelines:\n\n"
		for _, cf := range opts.ContextFiles {
			prompt += fmt.Sprintf("## %s\n\n%s\n\n", cf.Path, cf.Content)
		}
	}

	// --- Footer ---
	prompt += fmt.Sprintf("\nCurrent date: %s", date)
	prompt += fmt.Sprintf("\nCurrent working directory: %s", cwd)

	return prompt
}

// buildGuidelinesText returns the formatted bullet list of guidelines derived
// from the active tool set, extended by any caller-supplied extras.
func buildGuidelinesText(hasBash, hasRead, hasEdit, hasWrite bool, extra []string) string {
	seen := make(map[string]bool)
	var list []string
	add := func(g string) {
		if seen[g] {
			return
		}
		seen[g] = true
		list = append(list, g)
	}

	if hasBash {
		add("Use bash for file operations like ls, find, grep")
	}
	if hasRead && hasEdit {
		add("Use read to examine files before editing")
	}
	if hasEdit {
		add("Use edit for surgical, targeted changes to existing files")
	}
	if hasWrite {
		add("Use write only for new files or complete rewrites")
	}
	if hasEdit || hasWrite {
		add("When summarizing your actions, output plain text directly - do NOT use cat or bash to display what you did")
	}

	for _, g := range extra {
		g = strings.TrimSpace(g)
		if g != "" {
			add(g)
		}
	}

	// Always present.
	add("Be concise in your responses")
	add("Show file paths clearly when working with files")

	if len(list) == 0 {
		return "(none)"
	}
	lines := make([]string, len(list))
	for i, g := range list {
		lines[i] = "- " + g
	}
	return strings.Join(lines, "\n")
}
