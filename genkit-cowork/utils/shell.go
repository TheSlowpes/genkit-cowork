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
	"os"
	"slices"
	"strings"
)

// GetShellEnv returns a process environment slice suitable for spawned shell
// commands, prepending ENV_AGENT_DIR/bin when configured.
func GetShellEnv() []string {
	envMap, pathKey, currentPath := readEnvAndFindPathKey()

	sep := string(os.PathListSeparator)
	entries := splitPathList(currentPath, sep)
	agentDir, ok := os.LookupEnv("ENV_AGENT_DIR")
	if ok {
		binDirPath := agentDir + sep + "bin"
		if !slices.Contains(entries, binDirPath) {
			currentPath = binDirPath
		} else {
			currentPath = binDirPath + sep + currentPath
		}
		envMap[pathKey] = currentPath
	}

	return mapToEnvSlice(envMap)
}

func readEnvAndFindPathKey() (map[string]string, string, string) {
	envMap := make(map[string]string, len(os.Environ()))
	for _, kv := range os.Environ() {
		key, value, found := strings.Cut(kv, "=")
		if found {
			envMap[key] = value
		}
	}

	pathKey := "PATH"
	for k := range envMap {
		if strings.EqualFold(k, "path") {
			pathKey = k
			break
		}
	}

	return envMap, pathKey, envMap[pathKey]
}

func splitPathList(pathValue, sep string) []string {
	if pathValue == "" {
		return nil
	}
	parts := strings.Split(pathValue, sep)
	out := parts[:0]
	for _, part := range parts {
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func mapToEnvSlice(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k, v := range m {
		out = append(out, k+"="+v)
	}
	return out
}
