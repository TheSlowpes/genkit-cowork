package skills

import (
	"context"
	"sync"

	"github.com/firebase/genkit/go/core/api"
)

const provider = "skills"

type SkillRef interface {
	Name() string
}

type SkillName string

func (s SkillName) Name() string {
	return (string)(s)
}

type SkillDef struct {
	registry api.Registry
}

type SkillDefinition struct {
	Name        string   `json:"name,omitempty"`
	Description string   `json:"description,omitempty"`
	Location    string   `json:"location,omitempty"`
	Content     string   `json:"content,omitempty"`
	Scripts     []string `json:"scripts,omitempty"`
}

type Skill interface {
	Name() string
	Definition() SkillDefinition
	Register()
}

func DefineSkill(
	r api.Registry,
	name, description string,
)

type Skills struct {
	mu      sync.Mutex
	initted bool

	subSkills map[string]Skill
}
