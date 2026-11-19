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
	"encoding/binary"
	"fmt"

	"github.com/hybridgroup/yzma/pkg/mtmd"
)

func validateAudioPair(model map[string]ggufValue, header ggufHeader) error {
	projector := header.values
	if projector["general.architecture"].text != "clip" ||
		projector["clip.has_audio_encoder"] != (ggufValue{kind: 7, num: 1}) {
		return fmt.Errorf("GGUF projector has no audio encoder")
	}
	if len(header.tensors) == 0 || projector["split.no"].num != 0 || projector["split.count"].num > 1 {
		return fmt.Errorf("audio projector must be a single GGUF with tensors")
	}

	// Width alone does not distinguish Qwen3-ASR from unrelated decoders.
	projectorType := projector["clip.audio.projector_type"].text
	if projectorType == "" {
		projectorType = projector["clip.projector_type"].text
	}
	if projectorType != "qwen3a" || model["general.architecture"].text != "qwen3vl" ||
		model["general.basename"].text != "Qwen3-ASR" {
		return fmt.Errorf("audio input requires a Qwen3-ASR decoder and matching qwen3a projector")
	}
	width := model["qwen3vl.embedding_length"]
	if width.kind != 4 || width.num == 0 || width != projector["clip.audio.projection_dim"] {
		return fmt.Errorf("qwen3a projector width does not match the decoder")
	}
	return nil
}

func validateNativeAudio(mctx mtmd.Context) error {
	if !mtmd.SupportAudio(mctx) {
		return fmt.Errorf("loaded projector does not support audio input")
	}
	if rate := mtmd.GetAudioSampleRate(mctx); rate != audioSampleRate {
		return fmt.Errorf("loaded projector expects %d Hz audio, want %d Hz", rate, audioSampleRate)
	}
	return nil
}

func audioBitmap(pcm []byte) (mtmd.Bitmap, error) {
	if len(pcm) == 0 {
		return 0, fmt.Errorf("missing validated WAV samples")
	}
	samples := make([]float32, len(pcm)/pcmSampleBytes)
	for i := range samples {
		samples[i] = float32(int16(binary.LittleEndian.Uint16(pcm[i*pcmSampleBytes:]))) / 32768
	}
	bitmap := mtmd.BitmapInitFromAudio(uint64(len(samples)), &samples[0])
	if bitmap == 0 {
		return 0, fmt.Errorf("allocate audio bitmap")
	}
	return bitmap, nil
}
