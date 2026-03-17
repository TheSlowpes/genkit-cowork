package flows

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
)

// defaultToolDescriptions maps the built-in genkit-cowork tool names to
// one-line descriptions used when building the default system prompt.
var defaultToolDescriptions = map[string]string{
	"bash":  "Execute bash commands",
	"read":  "Read file contents",
	"edit":  "Make surgical edits to files (find exact text and replace)",
	"write": "Create or overwrite files",
}

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

// SystemPromptSkill describes a skill the agent can resolve for additional
// domain knowledge. Only the name and description appear in the system prompt
// listing; the full content is loaded on demand via the resolve-skill tool.
type SystemPromptSkill struct {
	Name        string
	Description string
}

// SystemPromptOptions configures the prompt produced by BuildSystemPrompt.
type SystemPromptOptions struct {
	// CustomPrompt replaces the auto-generated body when non-empty. The tools
	// list and guideline section are omitted, but AppendPrompt, ContextFiles,
	// Skills, and the date/cwd footer are still appended.
	CustomPrompt string

	// SelectedTools is the ordered list of tool names available to the agent.
	// Defaults to ["bash", "read", "edit", "write"] when nil or empty.
	SelectedTools []string

	// ToolSnippets provides one-line override descriptions for tools (including
	// custom tools not present in defaultToolDescriptions). Keyed by tool name.
	ToolSnippets map[string]string

	// Guidelines are additional bullet points appended after the
	// auto-generated guidelines. Duplicates are silently dropped.
	Guidelines []string

	// AppendPrompt is free-form text appended to the main prompt body before
	// context files, skills, and the date/cwd footer.
	AppendPrompt string

	// Cwd is the working directory reported in the prompt footer.
	// Defaults to the process working directory when empty.
	Cwd string

	// ContextFiles are project-specific instruction files embedded under a
	// "Project Context" heading at the end of the prompt.
	ContextFiles []ContextFile

	// Skills lists skills available to the agent. When the "read" tool is
	// included in the tool set, the skills are appended as a "Skills" section
	// so the agent knows what domain knowledge it can resolve on demand.
	Skills []SystemPromptSkill
}

// BuildSystemPrompt returns an ai.PromptFn that generates a system prompt for
// a genkit-cowork agent.
//
// The prompt includes:
//   - A tool list (built-in descriptions plus any ToolSnippets overrides).
//   - Usage guidelines derived automatically from the selected tool set,
//     extended by any caller-supplied Guidelines.
//   - An optional appended free-form section (AppendPrompt).
//   - An optional "Project Context" section from ContextFiles.
//   - An optional "Skills" section when the "read" tool is available.
//   - A footer with the current date and working directory.
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
		toolsList := buildToolsList(tools, opts.ToolSnippets)
		guidelinesText := buildGuidelinesText(hasBash, hasRead, hasEdit, hasWrite, opts.Guidelines)

		prompt = fmt.Sprintf(
			"You are a capable assistant that helps users get work done. "+
				"You can read and write files, run commands, and make targeted changes to existing content.\n\n"+
				"Available tools:\n%s\n\n"+
				"In addition to the tools above, you may have access to other custom tools "+
				"depending on the project.\n\n"+
				"Guidelines:\n%s",
			toolsList,
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

	// --- Skills (only when the read tool is available) ---
	if hasRead && len(opts.Skills) > 0 {
		prompt += "\n\n# Skills\n\nSkills you have access to " +
			"(use the resolve-skill tool to load the full content):\n\n"
		for _, skill := range opts.Skills {
			prompt += fmt.Sprintf("- %s: %s\n", skill.Name, skill.Description)
		}
	}

	// --- Footer ---
	prompt += fmt.Sprintf("\nCurrent date: %s", date)
	prompt += fmt.Sprintf("\nCurrent working directory: %s", cwd)

	return prompt
}

// buildToolsList returns the formatted bullet list of tool descriptions.
func buildToolsList(tools []string, snippets map[string]string) string {
	if len(tools) == 0 {
		return "(none)"
	}
	lines := make([]string, 0, len(tools))
	for _, name := range tools {
		desc := ""
		if snippets != nil {
			desc = snippets[name]
		}
		if desc == "" {
			desc = defaultToolDescriptions[name]
		}
		if desc == "" {
			desc = name
		}
		lines = append(lines, fmt.Sprintf("- %s: %s", name, desc))
	}
	return strings.Join(lines, "\n")
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
