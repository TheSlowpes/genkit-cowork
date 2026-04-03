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

package skills

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

const provider = "skills"

var _ api.Plugin = (*Skills)(nil)

// defaultSkillsDirs is the ordered list of directory paths tried when SkillsDir
// is not explicitly set. The first path that exists and is a directory wins.
var defaultSkillsDirs = []string{
	"./skills",
	"./SKILLS",
	"./.agent/skills",
	"./agent/skills",
	"./docs/skills",
	defaultGenkitSkillsDir(),
}

func defaultGenkitSkillsDir() string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ""
	}
	return filepath.Join(home, ".genkit", "skills")
}

// SkillDefinition holds the parsed contents of the SKILL.md frontmatter block.
// It represents a single discovered skill before its full content is loaded.
type SkillDefinition struct {
	// Name is the unique identifier of the skill (required).
	// Must be lowercase alphanumeric with hyphens, e.g. "my-skill".
	Name string `yaml:"name"`

	// Description is a human-readable summary of what the skill does (required).
	Description string `yaml:"description"`

	// License is an optional SPDX license identifier, e.g. "MIT".
	License string `yaml:"license,omitempty"`

	// Metadata holds any additional key/value pairs from the frontmatter.
	Metadata map[string]string `yaml:"metadata,omitempty"`

	// Files is a map of all files in the skill directory, keyed by filename.
	Files map[string]string

	// dir is the absolute path to the skill's directory on disk.
	// Unexported, set by the scanner, used by SkillTool to load the full body.
	dir string
}

// SkillToolOutput is the structured output returned by the resolve-skill tool.
// It combines the skill metadata with the full Markdown body so that both are
// available in the tool response Output field (avoiding PartText in Content,
// which is not supported by all model plugins).
type SkillToolOutput struct {
	// Definition holds the parsed SKILL.md frontmatter metadata.
	Definition *SkillDefinition `json:"definition"`
	// Body is the full Markdown content of the SKILL.md file (after frontmatter).
	Body string `json:"body"`
}

// Skills is the configuration for the plugin.
// Pass it to genkit.WithPlugins() to register all skills from a directory.
type Skills struct {
	// SkillsDir is the absolute or relative path to the directory containing skill subdirectories.
	// When empty, Init walks defaultSkillsDirs and picks the first existing one.
	SkillsDir string

	// AllowedSkills is an optional whitelist of skill names.
	// When non-empty, only skills whose Name appears in this slice are exposed
	// by SkillTool and ListSkills. All other discovered skills are hidden.
	AllowedSkills []string

	// cache holds the scanned list of skills.
	// Populated on Init and reused by ListSkills and SkillTool.
	cache []*SkillDefinition

	mu      sync.Mutex
	initted bool
}

// Name implements genkit.Plugin.
// Return the unique identifier for this plugin in Genkit's registry.
func (s *Skills) Name() string {
	return provider
}

// Init implements genkit.Plugin.
// Resolves the skills directory (trying defaultSkillsDirs when SkillsDir is
// unset), scans it for SKILL.md files, and populates the internal cache.
//
// Init does NOT panic when no directory is found; the cache simply stays empty
// and SkillTool will describe zero available skills.
//
// Panics only if SkillsDir is explicitly set but does not exist or is not a
// directory, or if the directory cannot be read.
// Individual skills that fail to parse are silently skipped.
func (s *Skills) Init(ctx context.Context) []api.Action {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.initted {
		panic("Skills plugin Init called more than once")
	}
	s.initted = true

	// Resolve skills directory.
	if s.SkillsDir == "" {
		for _, dir := range defaultSkillsDirs {
			if dir == "" {
				continue
			}
			info, err := os.Stat(dir)
			if err == nil && info.IsDir() {
				s.SkillsDir = dir
				break
			}
		}
	}

	// If still unresolved, leave the cache empty and return without error.
	if s.SkillsDir == "" {
		return nil
	}

	info, err := os.Stat(s.SkillsDir)
	if err != nil {
		panic("Skills plugin: skills dir does not exist or is unreadable: " + err.Error())
	}
	if !info.IsDir() {
		panic("Skills plugin: skills dir path exists but is not a directory")
	}

	skills, err := discoverSkills(s.SkillsDir)
	if err != nil {
		panic("Skills plugin: failed to read skills dir: " + err.Error())
	}

	s.cache = skills

	// Init returns nil actions — tools are registered by the caller via SkillTool.
	return nil
}

// ListSkills returns the cached skills visible to the plugin, respecting
// the AllowedSkills whitelist when it is non-empty.
func (s *Skills) ListSkills() []*SkillDefinition {
	return s.availableSkills()
}

// availableSkills returns the cached skills filtered by AllowedSkills.
// When AllowedSkills is empty, all discovered skills are returned.
func (s *Skills) availableSkills() []*SkillDefinition {
	if len(s.AllowedSkills) == 0 {
		return s.cache
	}
	allowed := make(map[string]bool, len(s.AllowedSkills))
	for _, name := range s.AllowedSkills {
		allowed[name] = true
	}
	filtered := make([]*SkillDefinition, 0, len(s.AllowedSkills))
	for _, skill := range s.cache {
		if allowed[skill.Name] {
			filtered = append(filtered, skill)
		}
	}
	return filtered
}

// buildSkillsDescription returns a multi-line string that enumerates all
// available skills so the model can see them directly in the tool description.
func (s *Skills) buildSkillsDescription() string {
	skills := s.availableSkills()

	var sb strings.Builder
	sb.WriteString("Resolve a skill by name to load its full Markdown content.\n\nAvailable skills:\n")
	if len(skills) == 0 {
		sb.WriteString("(none)\n")
	} else {
		for _, skill := range skills {
			fmt.Fprintf(&sb, "- %s: %s\n", skill.Name, skill.Description)
		}
	}
	return sb.String()
}

// SkillTool returns a single Genkit tool that exposes all available skills.
// The tool description is generated at registration time and lists every
// visible skill (name + description) so the model can decide which one to
// resolve without a separate listing call.
//
// The tool accepts a skill name and returns the full SKILL.md body together
// with the SkillDefinition metadata.
func (s *Skills) SkillTool(g *genkit.Genkit) ai.Tool {
	skills := s.availableSkills()
	description := s.buildSkillsDescription()

	return genkit.DefineMultipartTool(
		g,
		"resolve-skill",
		description,
		func(ctx *ai.ToolContext, name string) (*ai.MultipartToolResponse, error) {
			meta := findSkill(skills, name)
			if meta == nil {
				return nil, fmt.Errorf("no skill found with name %q", name)
			}

			body, err := loadSkillBody(meta.dir)
			if err != nil {
				return nil, fmt.Errorf("load skill body for %s: %w", name, err)
			}

			return &ai.MultipartToolResponse{
				Output: &SkillToolOutput{
					Definition: meta,
					Body:       body,
				},
			}, nil
		},
	)
}
