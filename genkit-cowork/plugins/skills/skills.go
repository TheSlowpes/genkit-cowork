package skills

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

const provider = "skills"

var _ api.Plugin = (*Skills)(nil)

// SkillDefinition holds the parsed contents of the SKILL.md frontmatter block.
// It represents a single discovered skill before its full content is loaded.
type SkillDefinition struct {
	// Name is the unique identifier or the skill (required).
	// Must be lowercase alphanumeric with hyphens, e.g. "my-skill".
	Name string `yaml:"name"`

	// Description is a human-readable summary of what the skill does (required)
	Description string `yaml:"description"`

	// License is an optional SPDX license identifier, e.g. "MIT".
	License string `yaml:"license,omitempty"`

	// Metadata holds any additional key/value pairs from the frontmatter.
	Metadata map[string]string `yaml:"metadata,omitempty"`

	// Files is a map of all files in the skill directory, keyed by filename.
	Files map[string]string

	// dir is the absolute path to the skill's directory on disk.
	// Unexported, set by the scanner, used by the ResolveAction to load the full body.
	dir string
}

// Skills is the configuration for the plugin.
// Pass it to genkit.WithPlugins() to register all skills from a directory.
type Skills struct {
	// SkillsDir is the absolute or relative path to the directory containing skill subdirectories.
	SkillsDir string

	// cache holds the scanned list of skills.
	// Populated on Init and reused by ListActions.
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
// Validades that the provided skills dir exists and performs the initial skill scan
// populating the cache so ListActions and ResolveAction are fast.
//
// Returns an error if the dir s missing or unreadable.
// Individual skills that fail to load during the scan will be skipped,
// but will not cause Init to fail.
func (s *Skills) Init(ctx context.Context) []api.Action {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.initted {
		panic("Skills plugin Init called more than once")
	}

	// Defaults to "./skills/" if not set, but can be overridden by the user.
	if s.SkillsDir == "" {
		s.SkillsDir = "./skills"
	}

	s.initted = true

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

	// Init returns nil actions — all actions are resolved dynamically via
	// ListActions and ResolveAction rather than being registered eagerly.
	return nil
}

// ListSkills returns the cached list of discovered skills.
func (s *Skills) ListSkills() []*SkillDefinition {
	return s.cache
}

type ListSkillsInput struct {
	Filter string `json:"filter,omitempty" jsonschema_description:"Optional filter string to only return skills whose name or description contains this substring."`
}

// ListSkillsTools is a wrapper around the ListSkills function
func (s *Skills) ListSkillsTool(g *genkit.Genkit) ai.Tool {
	return genkit.DefineTool(
		g,
		"list-skills",
		"List all available skills with their metadata.",
		func(ctx *ai.ToolContext, input ListSkillsInput) ([]*SkillDefinition, error) {
			if input.Filter == "" {
				return s.ListSkills(), nil
			}
			skills := s.ListSkills()
			filtered := make([]*SkillDefinition, 0, len(skills))
			for _, skill := range skills {
				if strings.Contains(skill.Name, input.Filter) {
					filtered = append(filtered, skill)
				}
			}

			if len(filtered) == 0 {
				return nil, fmt.Errorf("no skills found matching filter: %s", input.Filter)
			}

			return filtered, nil
		},
	)
}

func (s *Skills) ResolveSkillTool(g *genkit.Genkit) ai.Tool {
	return genkit.DefineMultipartTool(
		g,
		"resolve-skill",
		"Resolve a skill name to its full content, including loading any files in the skill directory.",
		func(ctx *ai.ToolContext, name string) (*ai.MultipartToolResponse, error) {
			meta := findSkill(s.cache, name)
			if meta == nil {
				return nil, fmt.Errorf("no skill found with name %s", name)
			}

			body, err := loadSkillBody(meta.dir)
			if err != nil {
				return nil, fmt.Errorf("load skill body for %s: %w", name, err)
			}

			textPart := ai.NewTextPart(body)

			response := &ai.MultipartToolResponse{
				Content: []*ai.Part{textPart},
				Output:  meta,
			}

			return response, nil
		},
	)
}
