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
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

const skillFileName = "SKILL.md"

// parseSkillMetadata reads a SKILL.md file at the given directory path,
// extracts the YAML frontmatter block, and returns a populated SkillDefinition.
//
// It expects the file to start with a "---" delimiter line, followed by
// YAML content, and closed by a second "---" line. Content after the
// closing delimiter is ignored at this stage (loaded later by ResolveSkill).
//
// Returns an error if:
//   - The file cannot be opened
//   - No valid frontmatter block is found
//   - The YAML is malformed
//   - Required fields (name, description) are missing
func parseSkillMetadata(dir string) (*SkillDefinition, error) {
	path := filepath.Join(dir, skillFileName)

	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()

	// Extract the raw YAML between the two "---" delimiters.
	raw, err := extractFrontMatter(file)
	if err != nil {
		return nil, fmt.Errorf("extract frontmatter from %s: %w", path, err)
	}

	var meta SkillDefinition
	if err := yaml.Unmarshal([]byte(raw), &meta); err != nil {
		return nil, fmt.Errorf("parse YAML in %s: %w", path, err)
	}

	// Validate required fields
	if err := validateSkillMeta(&meta, path); err != nil {
		return nil, err
	}

	// store the directory so ResolveSkill can load the full content later
	meta.dir = dir

	// store the full contents of the skill folder
	files, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read skill directory %s: %w", dir, err)
	}

	meta.Files = make(map[string]string, len(files))

	for _, file := range files {
		if file.IsDir() {
			subdirPath := filepath.Join(dir, file.Name())
			subFiles, err := os.ReadDir(subdirPath)
			if err != nil {
				continue
			}
			for _, subFile := range subFiles {
				meta.Files[filepath.Join(file.Name(), subFile.Name())] = filepath.Join(subdirPath, subFile.Name())
			}
		}
		meta.Files[file.Name()] = filepath.Join(dir, file.Name())
	}

	return &meta, nil
}

// extractFrontmatter scans a file and returns the raw YAML string
// between the opening and closing "---" delimiters.
func extractFrontMatter(file *os.File) (string, error) {
	scanner := bufio.NewScanner(file)

	if !scanner.Scan() || strings.TrimSpace(scanner.Text()) != "---" {
		return "", fmt.Errorf("missing opening '---' delimiter")
	}

	var sb strings.Builder

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "---" {
			return sb.String(), nil
		}
		sb.WriteString(line)
		sb.WriteByte('\n')
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("scanning file: %w", err)
	}

	return "", fmt.Errorf("frontmatter was never closed with '---' delimiter")
}

func validateSkillMeta(meta *SkillDefinition, path string) error {
	if strings.TrimSpace(meta.Name) == "" {
		return fmt.Errorf("skill at %s is missing required field: name", path)
	}
	if strings.TrimSpace(meta.Description) == "" {
		return fmt.Errorf("skill at %s is missing required field: description", path)
	}
	return nil
}

// loadSkillBody reads the full Markdown body of a SKILL.md file —
// Called lazily by ResolveAction, not during initial scanning.
func loadSkillBody(dir string) (string, error) {
	path := filepath.Join(dir, skillFileName)

	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	delimCount := 0
	var sb strings.Builder

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "---" {
			delimCount++
			continue
		}
		if delimCount >= 2 {
			sb.WriteString(line)
			sb.WriteByte('\n')
		}
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("scanning file: %w", err)
	}

	body := strings.TrimSpace(sb.String())
	if body == "" {
		return "", fmt.Errorf("skill body is empty in %s", path)
	}

	return body, nil
}
