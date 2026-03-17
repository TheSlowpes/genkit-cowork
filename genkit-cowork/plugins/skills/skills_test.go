package skills

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// mkSkill creates a minimal SKILL.md in a temporary directory and returns the
// skill subdirectory path.
func mkSkill(t *testing.T, root, name, description, body string) string {
	t.Helper()
	dir := filepath.Join(root, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkSkill: mkdir %s: %v", dir, err)
	}
	content := "---\nname: " + name + "\ndescription: " + description + "\n---\n" + body + "\n"
	if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte(content), 0o644); err != nil {
		t.Fatalf("mkSkill: write SKILL.md: %v", err)
	}
	return dir
}

// newInitedPlugin builds a Skills plugin pointing at dir and runs Init.
func newInitedPlugin(t *testing.T, dir string, allowed []string) *Skills {
	t.Helper()
	s := &Skills{SkillsDir: dir, AllowedSkills: allowed}
	s.Init(nil)
	return s
}

// ---------------------------------------------------------------------------
// discoverSkills / parseSkillMetadata
// ---------------------------------------------------------------------------

func TestDiscoverSkills_Basic(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "alpha", "Alpha skill", "Alpha body")
	mkSkill(t, root, "beta", "Beta skill", "Beta body")

	skills, err := discoverSkills(root)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(skills) != 2 {
		t.Fatalf("expected 2 skills, got %d", len(skills))
	}
}

func TestDiscoverSkills_SkipsInvalidSkill(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "valid", "Valid skill", "Valid body")

	// Create a directory without a SKILL.md
	if err := os.MkdirAll(filepath.Join(root, "no-skill"), 0o755); err != nil {
		t.Fatal(err)
	}

	skills, err := discoverSkills(root)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(skills) != 1 {
		t.Fatalf("expected 1 skill, got %d", len(skills))
	}
	if skills[0].Name != "valid" {
		t.Errorf("expected skill named 'valid', got %q", skills[0].Name)
	}
}

// ---------------------------------------------------------------------------
// Init — default directory resolution
// ---------------------------------------------------------------------------

func TestInit_ExplicitDir(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "my-skill", "Desc", "Body")

	s := &Skills{SkillsDir: root}
	s.Init(nil)

	if len(s.cache) != 1 {
		t.Fatalf("expected 1 cached skill, got %d", len(s.cache))
	}
}

func TestInit_NoDir_EmptyCache(t *testing.T) {
	// When no defaultSkillsDirs exist and SkillsDir is empty, Init should not
	// panic and should leave the cache empty.
	s := &Skills{}
	// Override defaultSkillsDirs to point nowhere for this test.
	orig := defaultSkillsDirs
	defaultSkillsDirs = []string{"/nonexistent/path/a", "/nonexistent/path/b"}
	defer func() { defaultSkillsDirs = orig }()

	s.Init(nil)

	if len(s.cache) != 0 {
		t.Fatalf("expected empty cache, got %d skills", len(s.cache))
	}
}

func TestInit_DefaultDirResolution(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "found-skill", "Found", "Body")

	orig := defaultSkillsDirs
	defaultSkillsDirs = []string{"/nonexistent", root}
	defer func() { defaultSkillsDirs = orig }()

	s := &Skills{}
	s.Init(nil)

	if s.SkillsDir != root {
		t.Errorf("expected SkillsDir=%q, got %q", root, s.SkillsDir)
	}
	if len(s.cache) != 1 {
		t.Fatalf("expected 1 skill, got %d", len(s.cache))
	}
}

func TestInit_PanicsOnSecondCall(t *testing.T) {
	root := t.TempDir()
	s := &Skills{SkillsDir: root}
	s.Init(nil)

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on second Init call, got none")
		}
	}()
	s.Init(nil)
}

// ---------------------------------------------------------------------------
// AllowedSkills filtering
// ---------------------------------------------------------------------------

func TestAvailableSkills_NoFilter(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "alpha", "Alpha", "Body")
	mkSkill(t, root, "beta", "Beta", "Body")

	s := newInitedPlugin(t, root, nil)
	skills := s.availableSkills()
	if len(skills) != 2 {
		t.Fatalf("expected 2 skills, got %d", len(skills))
	}
}

func TestAvailableSkills_WithFilter(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "alpha", "Alpha", "Body")
	mkSkill(t, root, "beta", "Beta", "Body")
	mkSkill(t, root, "gamma", "Gamma", "Body")

	s := newInitedPlugin(t, root, []string{"alpha", "gamma"})
	skills := s.availableSkills()
	if len(skills) != 2 {
		t.Fatalf("expected 2 skills after filter, got %d", len(skills))
	}
	for _, sk := range skills {
		if sk.Name != "alpha" && sk.Name != "gamma" {
			t.Errorf("unexpected skill %q in filtered result", sk.Name)
		}
	}
}

func TestAvailableSkills_FilterUnknownName(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "alpha", "Alpha", "Body")

	s := newInitedPlugin(t, root, []string{"nonexistent"})
	skills := s.availableSkills()
	if len(skills) != 0 {
		t.Fatalf("expected 0 skills, got %d", len(skills))
	}
}

func TestListSkills_RespectsAllowedSkills(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "alpha", "Alpha", "Body")
	mkSkill(t, root, "beta", "Beta", "Body")

	s := newInitedPlugin(t, root, []string{"beta"})
	skills := s.ListSkills()
	if len(skills) != 1 {
		t.Fatalf("expected 1 skill from ListSkills, got %d", len(skills))
	}
	if skills[0].Name != "beta" {
		t.Errorf("expected 'beta', got %q", skills[0].Name)
	}
}

// ---------------------------------------------------------------------------
// buildSkillsDescription
// ---------------------------------------------------------------------------

func TestBuildSkillsDescription_ContainsSkillNames(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "alpha", "Does alpha things", "Body")
	mkSkill(t, root, "beta", "Does beta things", "Body")

	s := newInitedPlugin(t, root, nil)
	desc := s.buildSkillsDescription()

	for _, want := range []string{"alpha", "Does alpha things", "beta", "Does beta things"} {
		if !strings.Contains(desc, want) {
			t.Errorf("description missing %q\n\ndescription:\n%s", want, desc)
		}
	}
}

func TestBuildSkillsDescription_EmptySkills(t *testing.T) {
	s := &Skills{}
	orig := defaultSkillsDirs
	defaultSkillsDirs = []string{}
	defer func() { defaultSkillsDirs = orig }()
	s.Init(nil)

	desc := s.buildSkillsDescription()
	if !strings.Contains(desc, "(none)") {
		t.Errorf("expected '(none)' in description for zero skills, got:\n%s", desc)
	}
}

func TestBuildSkillsDescription_RespectsAllowedSkills(t *testing.T) {
	root := t.TempDir()
	mkSkill(t, root, "alpha", "Alpha desc", "Body")
	mkSkill(t, root, "beta", "Beta desc", "Body")

	s := newInitedPlugin(t, root, []string{"alpha"})
	desc := s.buildSkillsDescription()

	if !strings.Contains(desc, "alpha") {
		t.Errorf("expected 'alpha' in description")
	}
	if strings.Contains(desc, "beta") {
		t.Errorf("expected 'beta' to be absent from description (not in AllowedSkills)")
	}
}

