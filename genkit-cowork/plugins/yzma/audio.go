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
)

const (
	maxAudioBytes   = 1 << 20
	maxAudioSeconds = 30
	audioSampleRate = 16000
	pcmSampleBytes  = 2
)

func decodeAudioPart(mime, payload string, budget int) (MediaInputPart, error) {
	data, err := decodeInlineBytes(payload, min(budget, maxAudioBytes))
	if err != nil {
		return MediaInputPart{}, err
	}
	pcm, err := wavPCM16(data)
	if err != nil {
		return MediaInputPart{}, err
	}
	return MediaInputPart{MIMEType: mime, Data: data, pcm16: pcm}, nil
}

// wavPCM16 validates the RIFF container and returns a view of its PCM bytes.
// Do not pass untrusted container headers to the native audio decoder.
func wavPCM16(data []byte) ([]byte, error) {
	le := binary.LittleEndian
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" ||
		uint64(le.Uint32(data[4:8])) != uint64(len(data)-8) {
		return nil, fmt.Errorf("invalid RIFF/WAVE header or size")
	}
	var pcm []byte
	hasFormat := false
	for rest := data[12:]; len(rest) > 0; {
		if len(rest) < 8 {
			return nil, fmt.Errorf("truncated WAV chunk header")
		}
		size := uint64(le.Uint32(rest[4:8]))
		padded := size + size%2
		if padded > uint64(len(rest)-8) {
			return nil, fmt.Errorf("truncated WAV chunk")
		}
		chunk := rest[8 : 8+int(size)]
		switch string(rest[:4]) {
		case "fmt ":
			if hasFormat {
				return nil, fmt.Errorf("duplicate WAV fmt chunk")
			}
			if !isMonoPCM16(chunk) {
				return nil, fmt.Errorf("WAV must be mono 16 kHz signed 16-bit PCM")
			}
			hasFormat = true
		case "data":
			if !hasFormat || pcm != nil || len(chunk) == 0 || len(chunk)%pcmSampleBytes != 0 {
				return nil, fmt.Errorf("invalid WAV data chunk: expected one non-empty PCM chunk after fmt")
			}
			if len(chunk)/pcmSampleBytes > audioSampleRate*maxAudioSeconds {
				return nil, fmt.Errorf("WAV duration exceeds the %d-second limit", maxAudioSeconds)
			}
			pcm = chunk
		}
		rest = rest[8+int(padded):]
	}
	if pcm == nil {
		return nil, fmt.Errorf("missing WAV data chunk")
	}
	return pcm, nil
}

func isMonoPCM16(format []byte) bool {
	le := binary.LittleEndian
	if len(format) != 16 && (len(format) != 18 || le.Uint16(format[16:]) != 0) {
		return false
	}
	return le.Uint16(format[0:2]) == 1 && // WAVE_FORMAT_PCM
		le.Uint16(format[2:4]) == 1 &&
		le.Uint32(format[4:8]) == audioSampleRate &&
		le.Uint32(format[8:12]) == audioSampleRate*pcmSampleBytes &&
		le.Uint16(format[12:14]) == pcmSampleBytes &&
		le.Uint16(format[14:16]) == 8*pcmSampleBytes
}
