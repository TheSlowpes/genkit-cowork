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

package memory

import (
	"context"
	"testing"
)

func TestDefaultPreferenceOperator_CRUD(t *testing.T) {
	op := NewDefaultPreferenceOperator()
	ctx := context.Background()

	created, err := op.SavePreference(ctx, "tenant-1", PreferenceRecord{
		Key:        "timezone",
		Value:      "America/Sao_Paulo",
		Source:     PreferenceSourceExplicit,
		Confidence: 1,
	})
	if err != nil {
		t.Fatalf("SavePreference(create) error = %v", err)
	}
	if created.PreferenceID == "" {
		t.Fatal("expected generated preferenceID")
	}

	loaded, err := op.LoadPreference(ctx, "tenant-1", created.PreferenceID)
	if err != nil {
		t.Fatalf("LoadPreference() error = %v", err)
	}
	if loaded == nil || loaded.Key != "timezone" {
		t.Fatalf("LoadPreference() = %+v, want key timezone", loaded)
	}

	updated, err := op.SavePreference(ctx, "tenant-1", PreferenceRecord{
		PreferenceID: created.PreferenceID,
		Key:          "timezone",
		Value:        "UTC",
		Source:       PreferenceSourceExplicit,
		Status:       PreferenceStatusActive,
	})
	if err != nil {
		t.Fatalf("SavePreference(update) error = %v", err)
	}
	if updated.Value != "UTC" {
		t.Fatalf("updated value = %q, want UTC", updated.Value)
	}

	list, err := op.ListPreferences(ctx, "tenant-1", PreferenceFilter{Source: PreferenceSourceExplicit, Status: PreferenceStatusActive})
	if err != nil {
		t.Fatalf("ListPreferences() error = %v", err)
	}
	if len(list) != 1 {
		t.Fatalf("len(ListPreferences()) = %d, want 1", len(list))
	}

	if err := op.DeletePreference(ctx, "tenant-1", created.PreferenceID); err != nil {
		t.Fatalf("DeletePreference() error = %v", err)
	}
	afterDelete, err := op.LoadPreference(ctx, "tenant-1", created.PreferenceID)
	if err != nil {
		t.Fatalf("LoadPreference(after delete) error = %v", err)
	}
	if afterDelete != nil {
		t.Fatalf("LoadPreference(after delete) = %+v, want nil", afterDelete)
	}
}

func TestFilePreferenceOperator_SaveAndFilter(t *testing.T) {
	op := NewFilePreferenceOperator(t.TempDir())
	ctx := context.Background()

	if _, err := op.SavePreference(ctx, "tenant-1", PreferenceRecord{
		Key:        "tone",
		Value:      "concise",
		Source:     PreferenceSourceImplicit,
		Confidence: 0.85,
	}); err != nil {
		t.Fatalf("SavePreference() error = %v", err)
	}

	if _, err := op.SavePreference(ctx, "tenant-1", PreferenceRecord{
		Key:        "timezone",
		Value:      "UTC",
		Source:     PreferenceSourceExplicit,
		Confidence: 1,
	}); err != nil {
		t.Fatalf("SavePreference() error = %v", err)
	}

	implicit, err := op.ListPreferences(ctx, "tenant-1", PreferenceFilter{Source: PreferenceSourceImplicit})
	if err != nil {
		t.Fatalf("ListPreferences(implicit) error = %v", err)
	}
	if len(implicit) != 1 || implicit[0].Key != "tone" {
		t.Fatalf("implicit preferences = %+v, want one key=tone", implicit)
	}
}

func TestPreferenceOperator_TenantIsolation(t *testing.T) {
	op := NewDefaultPreferenceOperator()
	ctx := context.Background()

	created, err := op.SavePreference(ctx, "tenant-a", PreferenceRecord{Key: "lang", Value: "pt", Source: PreferenceSourceExplicit})
	if err != nil {
		t.Fatalf("SavePreference() error = %v", err)
	}

	got, err := op.LoadPreference(ctx, "tenant-b", created.PreferenceID)
	if err != nil {
		t.Fatalf("LoadPreference() error = %v", err)
	}
	if got != nil {
		t.Fatalf("cross-tenant LoadPreference() = %+v, want nil", got)
	}
}
