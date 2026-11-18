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

package yzma

import (
	"fmt"
	"strings"
)

// Automatically pair only families whose decoder and projector metadata we
// know how to match. Other families require an explicit definition.
func discoverProjector(filename string, files map[string]ggufCandidate, model *ModelDefinition) error {
	values := files[filename].header.values
	if values["general.architecture"].text != "llama" || values["general.basename"].text != "SmolVLM" {
		return nil
	}
	if model.Type != ModelTypeChat {
		return fmt.Errorf("SmolVLM requires a chat template; register it with DefineModel")
	}
	var matching []string
	for name, candidate := range files {
		projection := candidate.header.values
		if projection["clip.projector_type"].text != "idefics3" ||
			projection["general.name"].text == "" ||
			projection["general.name"].text != values["general.name"].text ||
			validateVisionPair(values, candidate.header) != nil {
			continue
		}
		if strings.EqualFold(name, "mmproj-"+filename) {
			model.Projector = &ProjectorOptions{GGUFPath: candidate.path, Images: true}
			return nil
		}
		matching = append(matching, candidate.path)
	}
	if len(matching) != 1 {
		return fmt.Errorf("found %d compatible vision projectors; set ModelDefinition.Projector explicitly", len(matching))
	}
	model.Projector = &ProjectorOptions{GGUFPath: matching[0], Images: true}
	return nil
}

func validateVisionPair(model map[string]ggufValue, header ggufHeader) error {
	projector := header.values
	if projector["general.architecture"].text != "clip" ||
		projector["clip.has_vision_encoder"] != (ggufValue{kind: 7, num: 1}) {
		return fmt.Errorf("GGUF projector has no vision encoder")
	}
	if len(header.tensors) == 0 || projector["split.no"].num != 0 || projector["split.count"].num > 1 {
		return fmt.Errorf("vision projector must be a single GGUF with tensors")
	}
	if projector["clip.projector_type"].text == "idefics3" {
		modelWidth := model["llama.embedding_length"]
		if modelWidth.kind != 4 || modelWidth.num == 0 ||
			modelWidth != projector["clip.vision.projection_dim"] {
			return fmt.Errorf("idefics3 projector width does not match the decoder")
		}
		if a, b := model["general.name"].text, projector["general.name"].text; a != "" && b != "" && a != b {
			return fmt.Errorf("idefics3 projector is for %q, not %q", b, a)
		}
	}
	return nil
}
