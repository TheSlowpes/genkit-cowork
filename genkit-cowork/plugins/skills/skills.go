package skills

import (
	"context"
	"os"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

const provider = "skills"

var _ api.DynamicPlugin = (*Skills)(nil)

// SkillDefinition holds the parsed contents of the SKILL.md frontmatter block.
// It represents a ingle discovered skill before its full content is loaded.
type SkillDefinition struct {
	// Name is the nique identifier r the skill (required).
	// Must be lowercase lphanumeric ith hyphens, e.g. "my-skill".
	Name string `yaml:"name"`

	// Description s a human-readable summary of what the skill does (required)
	// Used as the tool description n Genkit's registry.
	Description string `yaml:"description"`

	// License is an optional SPDX license identifier, e.g. "MIT".
	License string `yaml:"license,omitempty"`

	// Metadata holds any additional key/value pairs from the frontmatter.
	Metadata map[string]string `yaml:"metadata,omitempty"`

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

// ListActions implements api.DynamicPlugin.
// Returns a descriptor for every skill currently in the cache.
// Each skill is advertised as a tool action — no full Markdown is loaded here.
func (s *Skills) ListActions(ctx context.Context) []api.ActionDesc {
	descs := make([]api.ActionDesc, 0, len(s.cache))

	for _, meta := range s.cache {
		descs = append(descs, api.ActionDesc{
			Key:         api.NewKey(api.ActionTypeTool, provider, meta.Name),
			Description: meta.Description,
		})
	}

	return descs
}

// ResolveAction implements api.DynamicPlugin.
// Called by Genkit's registry when a tool with the given name is needed.
// Only at this point is the full SKILL.md body read from disk and the
// ai.Tool defined and returned.
//
// Returns nil if the requested action type is not ActionTypeTool, or if
// no skill with the given name exists in the cache.
func (s *Skills) ResolveAction(atype api.ActionType, name string) api.Action {
	if atype != api.ActionTypeTool {
		return nil
	}

	meta := findSkill(s.cache, name)
	if meta == nil {
		return nil
	}

	body, err := loadSkillBody(meta.dir)
	if err != nil {
		return nil
	}

	skillName := meta.Name
	skillBody := body

}
