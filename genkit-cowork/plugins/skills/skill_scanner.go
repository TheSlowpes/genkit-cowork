// Copyright 2025 Google LLC
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
