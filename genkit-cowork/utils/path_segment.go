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

package utils

import (
	"fmt"
	"strings"
)

// ValidatePathSegment ensures that a value used as a single filesystem path
// component is safe. It rejects empty strings, values that contain path
// separators ('/' or '\\'), and dot-directory references ('.' and '..') to
// prevent path-traversal attacks when the value is interpolated into a
// filepath.Join call.
func ValidatePathSegment(name, value string) error {
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("%s is required", name)
	}
	if strings.ContainsAny(value, `/\`) {
		return fmt.Errorf("%s must not contain path separators", name)
	}
	if value == "." || value == ".." {
		return fmt.Errorf("%s must not be a dot-directory reference", name)
	}
	return nil
}
