package utils

import (
	"os"
	"slices"
	"strings"
)

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
