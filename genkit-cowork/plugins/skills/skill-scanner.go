package skills

import (
	"fmt"
	"os"
)

// discoverSkills walks the top level of dir, attempts to parse a SKILL.md
// in each subdirectory, and returns all successfully loaded skills.
//
// Subdirectories with missing or invalid SKILL.md files are logged and skipped —
// they will never cause the plugin to fail on startup.
//
// Returns an error only if dir itself cannot be read.
func discoverSkills(dir string) ([]*SkillDefinition, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read skills directory %s: %w", dir, err)
	}

	var skills []*SkillDefinition

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		skillDir := fmt.Sprintf("%s/%s", dir, entry.Name())

		meta, err := parseSkillMetadata(skillDir)
		if err != nil {
			continue
		}

		skills = append(skills, meta)
	}

	return skills, nil
}

// findSkill searches a slice of SkillDefinition for one matching the given name
// Returns nil if no match is found.
func findSkill(skills []*SkillDefinition, name string) *SkillDefinition {
	for _, skill := range skills {
		if skill.Name == name {
			return skill
		}
	}
	return nil
}
