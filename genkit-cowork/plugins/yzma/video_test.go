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
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/hybridgroup/yzma/pkg/llama"
)

func mp4Part(data []byte) *ai.Part {
	return ai.NewMediaPart(videoMIME, "data:video/mp4;base64,"+base64.StdEncoding.EncodeToString(data))
}

func mp4ContainerFixture() []byte {
	box := func(kind string, body []byte) []byte {
		size := binary.BigEndian.AppendUint32(nil, uint32(8+len(body)))
		return append(append(size, kind...), body...)
	}
	header := binary.BigEndian.AppendUint32(make([]byte, 12), 1000)
	header = binary.BigEndian.AppendUint32(header, 1000)
	// Complete fast-start container with no H.264 track.
	return bytes.Join([][]byte{box("ftyp", []byte("isom0000")), box("moov", box("mvhd", header)),
		box("mdat", []byte("media"))}, nil)
}

func TestCustomVideoOperatorReceivesValidatedBytesWithoutFFmpeg(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	mp4 := mp4ContainerFixture()
	called := false
	local, g := initDiscoveryTest(t, t.TempDir(), &testVisionOperator{
		testModelOperator: &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) {
			return 42, nil
		}},
		loadProjector: func(context.Context, llama.Model, ModelDefinition) (ProjectorHandle, error) { return 7, nil },
		freeProjector: func(context.Context, ProjectorHandle) error { return nil },
		generate: func(_ context.Context, _ llama.Model, _ ProjectorHandle, input MediaGenerationInput) (TextGenerationResult, error) {
			called = true
			if input.Type != ModelTypeChat || input.MaxOutputTokens != 40 {
				t.Errorf("settings = %+v", input.GenerationSettings)
			}
			part := input.Messages[0].Parts[0]
			if part.MIMEType != videoMIME || !bytes.Equal(part.Data, mp4) || len(part.VideoFrames) != 0 {
				t.Errorf("received part = %+v, want original MP4 and no decoded frames", part)
			}
			return TextGenerationResult{Text: "Video received"}, nil
		},
	})
	def := testModelDefinition()
	def.Projector = &ProjectorOptions{GGUFPath: "video.gguf", Video: true}
	def.GenerationDefaults = &GenerateContentConfig{MaxOutputTokens: ptr(40)}
	model := local.DefineModel(g, def, nil)
	_, err := genkit.Generate(t.Context(), g, ai.WithModel(model), ai.WithMessages(ai.NewUserMessage(mp4Part(mp4))))
	if err != nil || !called {
		t.Fatalf("Generate error = %v, called = %v", err, called)
	}
}

type videoDecoderFunc func(context.Context, []byte) ([]image.Image, error)

func (f videoDecoderFunc) DecodeVideo(ctx context.Context, data []byte) ([]image.Image, error) {
	return f(ctx, data)
}

func TestNativeVideoPreparationRunsAfterContainerValidationBeforeWeightLoad(t *testing.T) {
	const badContainer = "not an MP4"
	badTrack := errors.New("video track rejected by decoder")
	calls := 0
	op := defaultModelOperator{videoDecoder: videoDecoderFunc(func(context.Context, []byte) ([]image.Image, error) {
		calls++
		return nil, badTrack
	})}
	// No model resources or load gate exist. We must reject both requests
	// before attempting to acquire the model.
	g := &generator{operator: op, model: ModelDefinition{Projector: &ProjectorOptions{Video: true}}}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	req := ai.NewModelRequest(nil, ai.NewUserMessage(mp4Part([]byte(badContainer))))
	if _, err := g.generateMedia(ctx, req, nil); err == nil || !strings.Contains(err.Error(), "invalid MP4") || calls != 0 {
		t.Fatalf("invalid container produced error %v and %d decoder calls", err, calls)
	}
	req = ai.NewModelRequest(nil, ai.NewUserMessage(mp4Part(mp4ContainerFixture())))
	if _, err := g.generateMedia(ctx, req, nil); !errors.Is(err, badTrack) || calls != 1 {
		t.Fatalf("invalid track produced error %v and %d decoder calls", err, calls)
	}
}

func TestVideoDecoderBounds(t *testing.T) {
	data := []byte("validated container")
	frame := image.NewRGBA(image.Rect(0, 0, videoSampleEdge, videoSampleEdge))
	for _, tc := range []struct {
		name   string
		frames []image.Image
		want   string
	}{
		{"valid", []image.Image{frame}, ""},
		{"empty", nil, "between 1 and 4"},
		{"too many", []image.Image{frame, frame, frame, frame, frame}, "between 1 and 4"},
		{"too large", []image.Image{image.NewRGBA(image.Rect(0, 0, 257, 256))}, "256x256"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			op := videoDecoderFunc(func(ctx context.Context, got []byte) ([]image.Image, error) {
				if _, ok := ctx.Deadline(); !ok || !bytes.Equal(got, data) {
					t.Error("decoder did not receive bytes and a bounded context")
				}
				return tc.frames, nil
			})
			got, err := decodeVideoSamples(t.Context(), op, data)
			if tc.want == "" {
				if err != nil || len(got) != 1 || got[0] != frame {
					t.Fatalf("decode = (%v, %v), want one frame", got, err)
				}
			} else if got != nil || err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("decode = (%v, %v), want %q", got, err, tc.want)
			}
		})
	}
}

func TestVideoRejectsInvalidGenkitInputBeforeLoad(t *testing.T) {
	local, g := initDiscoveryTest(t, t.TempDir(), &testVisionOperator{testModelOperator: &testModelOperator{
		load: func(context.Context, ModelOptions) (llama.Model, error) {
			t.Fatal("invalid video request must not load weights")
			return 0, nil
		},
	}})
	def := testModelDefinition()
	def.Projector = &ProjectorOptions{GGUFPath: "video.gguf", Video: true}
	model := local.DefineModel(g, def, nil)

	for _, tc := range []struct {
		name, want string
		part       *ai.Part
	}{
		{"remote", "inline base64", ai.NewMediaPart(videoMIME, "https://example.org/clip.mp4")},
		{"file", "inline base64", ai.NewMediaPart(videoMIME, "file:///tmp/clip.mp4")},
		{"format", "unsupported media MIME", ai.NewMediaPart("video/webm", "data:video/webm;base64,AAAA")},
		{"mismatch", "content type does not match", ai.NewMediaPart("video/webm", mp4Part([]byte("clip")).Text)},
		{"standalone image disabled", "unsupported media MIME", pngPart([]byte("image"))},
		{"audio disabled", "unsupported media MIME", ai.NewMediaPart("audio/wav", "data:audio/wav;base64,AAAA")},
		{"malformed", "invalid MP4", mp4Part([]byte("not an MP4"))},
		{"oversize", "4194304-byte limit", mp4Part(make([]byte, maxVideoBytes+1))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := genkit.Generate(t.Context(), g, ai.WithModel(model), ai.WithMessages(ai.NewUserMessage(tc.part)))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Generate error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestConstrainedMediaRejectsAudioAndVideoBeforeDecode(t *testing.T) {
	local, g := initDiscoveryTest(t, t.TempDir(), &testConstrainedMediaOperator{testVisionOperator: &testVisionOperator{
		testModelOperator: &testModelOperator{load: func(context.Context, ModelOptions) (llama.Model, error) {
			t.Fatal("constrained audio or video must not load weights")
			return 0, nil
		}},
	}})
	def := testModelDefinition()
	def.Projector = &ProjectorOptions{GGUFPath: "manual.gguf", Images: true, Audio: true, Video: true}
	model := local.DefineModel(g, def, nil)
	for _, mime := range []string{"audio/wav", videoMIME} {
		t.Run(mime, func(t *testing.T) {
			// Invalid base64 verifies the constraint is rejected before decoding.
			part := ai.NewMediaPart(mime, "data:"+mime+";base64,%%%%")
			_, err := model.Generate(t.Context(), &ai.ModelRequest{
				Messages: []*ai.Message{ai.NewUserMessage(part)},
				Output: &ai.ModelOutputConfig{Constrained: true, Format: ai.OutputFormatJSON, Schema: map[string]any{
					"type": "object", "properties": map[string]any{}, "additionalProperties": false,
				}},
			}, nil)
			if err == nil || !strings.Contains(err.Error(), "constrained media output supports only image input") {
				t.Fatalf("Generate error = %v, want image-only constraint error", err)
			}
		})
	}
}

func TestVideoProbeLimits(t *testing.T) {
	for _, tc := range []struct {
		name, codec, frames, want  string
		width, packets             int
		duration, pts, endDuration uint64
	}{
		{"within limits", "h264", "300", "", 1024, 300, 10000, 9999, 1},
		{"unsupported codec", "hevc", "300", "H.264", 1024, 300, 10000, 9999, 1},
		{"too many frames", "h264", "301", "source frames", 1024, 301, 10000, 9999, 1},
		{"understated frames", "h264", "299", "source frames", 1024, 300, 10000, 9999, 1},
		{"unknown frames", "h264", "N/A", "source frames", 1024, 300, 10000, 9999, 1},
		{"long track", "h264", "300", "duration", 1024, 300, 10001, 9999, 1},
		{"late frame", "h264", "300", "timeline", 1024, 300, 10000, 10001, 1},
		{"late frame end", "h264", "300", "timeline", 1024, 300, 10000, 9999, 2},
		{"oversize dimensions", "h264", "300", "dimensions", 1025, 300, 10000, 9999, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result := videoProbe{
				Streams: []videoTrack{{Codec: tc.codec, Width: tc.width, Height: 256, Frames: tc.frames, TimeBase: "1/1000", Duration: tc.duration}},
				Packets: make([]videoPacket, tc.packets),
			}
			for i := range result.Packets {
				result.Packets[i] = videoPacket{PTS: ptr(uint64(i * 30)), Duration: 30}
			}
			result.Packets[len(result.Packets)-1] = videoPacket{PTS: &tc.pts, Duration: tc.endDuration}
			got, err := result.frameCount()
			if tc.want == "" {
				if err != nil || got != maxVideoFrames {
					t.Fatalf("frameCount = (%d, %v), want (%d, nil)", got, err, maxVideoFrames)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("frameCount error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestMP4ContainerLimits(t *testing.T) {
	box := func(kind string, body []byte) []byte {
		size := binary.BigEndian.AppendUint32(nil, uint32(8+len(body)))
		return append(append(size, kind...), body...)
	}
	moov := func(duration uint32) []byte {
		header := binary.BigEndian.AppendUint32(make([]byte, 12), 1000)
		header = binary.BigEndian.AppendUint32(header, duration)
		return box("moov", box("mvhd", header))
	}
	ftyp, mdat := box("ftyp", []byte("isom0000")), box("mdat", []byte("video data"))
	for _, tc := range []struct {
		name, want string
		boxes      [][]byte
	}{
		{"within duration", "", [][]byte{ftyp, moov(10000), mdat}},
		{"long movie", "duration", [][]byte{ftyp, moov(10001), mdat}},
		{"not fast-start", "fast-start", [][]byte{ftyp, mdat, moov(10000)}},
		{"truncated box", "incomplete box", [][]byte{ftyp, moov(10000), mdat[:len(mdat)-1]}},
		{"fragment", "fragmented MP4", [][]byte{ftyp, moov(10000), box("moof", nil), mdat}},
		{"QuickTime", "unrecognized MP4 file type", [][]byte{box("ftyp", []byte("qt  0000")), moov(10000), mdat}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateMP4Boxes(bytes.Join(tc.boxes, nil))
			if (tc.want == "" && err != nil) || (tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want))) {
				t.Errorf("validateMP4Boxes error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestVideoSamplesDisplayOrderAndRejectsCountMismatch(t *testing.T) {
	const frameBytes = 3 * videoSampleEdge * videoSampleEdge
	var stream []byte
	for i := range 8 {
		stream = append(stream, bytes.Repeat([]byte{byte(i)}, frameBytes)...)
	}
	frames, err := readVideoFrames(bytes.NewReader(stream), 8)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint32{0, 2, 4, 7}
	if len(frames) != len(want) {
		t.Fatalf("sampled %d frames, want %d", len(frames), len(want))
	}
	for i, frame := range frames {
		r, _, _, alpha := frame.At(0, 0).RGBA()
		if r>>8 != want[i] || alpha != 65535 {
			t.Errorf("sample %d = red %d, alpha %d; want red %d, alpha 65535", i, r>>8, alpha, want[i])
		}
	}
	for _, count := range []int{7, 9} {
		if _, err := readVideoFrames(bytes.NewReader(stream), count); err == nil {
			t.Errorf("readVideoFrames accepted 8 actual frames with a declared count of %d", count)
		}
	}
}

func TestVideoRejectsIncompatibleProjector(t *testing.T) {
	const modelName = "SmolVLM 256M Instruct"
	model := map[string]ggufValue{
		"general.architecture":   {kind: ggufString, text: "llama"},
		"general.basename":       {kind: ggufString, text: "SmolVLM"},
		"general.name":           {kind: ggufString, text: modelName},
		"llama.embedding_length": {kind: 4, num: 576},
	}
	path := filepath.Join(t.TempDir(), "projector.gguf")
	values := map[string]ggufValue{
		"general.architecture":       {kind: ggufString, text: "clip"},
		"general.name":               {kind: ggufString, text: modelName},
		"clip.projector_type":        {kind: ggufString, text: "idefics3"},
		"clip.has_vision_encoder":    {kind: 7, num: 1},
		"clip.vision.projection_dim": {kind: 4, num: 576},
	}
	check := func(want string) {
		t.Helper()
		writeGGUF(t, filepath.Dir(path), filepath.Base(path), values, "vision.weight")
		h, err := readGGUFHeader(t.Context(), path)
		if err != nil {
			t.Fatal(err)
		}
		err = validateVideoPair(model, h)
		if (want == "" && err != nil) || (want != "" && (err == nil || !strings.Contains(err.Error(), want))) {
			t.Errorf("validateVideoPair error = %v, want %q", err, want)
		}
	}
	check("")
	values["clip.vision.projection_dim"] = ggufValue{kind: 4, num: 512}
	check("width does not match")
	values["clip.projector_type"] = ggufValue{kind: ggufString, text: "unknown"}
	check("matching idefics3 projector")
}

func TestNativeVideoGeneration(t *testing.T) {
	modelPath, projectorPath := os.Getenv("YZMA_TEST_MODEL"), os.Getenv("YZMA_TEST_PROJECTOR")
	if modelPath == "" || projectorPath == "" {
		t.Skip("set YZMA_TEST_MODEL and YZMA_TEST_PROJECTOR to the SmolVLM 256M test pair")
	}
	local := &Yzma{ModelsDir: filepath.Dir(modelPath)}
	g := genkit.Init(t.Context(), genkit.WithPlugins(local))
	t.Cleanup(func() {
		if err := local.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	model := local.DefineModel(g, ModelDefinition{
		Name: "local-video", Type: ModelTypeChat,
		Loading:   ModelOptions{GGUFPath: modelPath, NGPULayers: ptr(int32(0))},
		Projector: &ProjectorOptions{GGUFPath: projectorPath, Video: true},
		Context:   ContextOptions{ContextSize: ptr(uint32(4096)), Threads: ptr(int32(2)), BatchThreads: ptr(int32(2))},
	}, nil)
	for _, tc := range []struct{ first, last string }{{"red", "blue"}, {"blue", "red"}} {
		video := nativeTestVideo(t, tc.first, tc.last)
		response, err := genkit.Generate(t.Context(), g, ai.WithModel(model),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Watch this clip: "), mp4Part(video),
				ai.NewTextPart(" What color is the first frame?"))),
			ai.WithConfig(&GenerateContentConfig{MaxOutputTokens: ptr(32), Temperature: ptr(0.0)}),
		)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("first color %s: %q", tc.first, response.Text())
		answer := strings.ToLower(response.Text())
		firstAt, lastAt := strings.Index(answer, tc.first), strings.Index(answer, tc.last)
		if firstAt < 0 || (lastAt >= 0 && lastAt < firstAt) || response.Usage.InputTokens < 100 {
			t.Errorf("expected %s first and vision token usage, got %q with %+v", tc.first, response.Text(), response.Usage)
		}
	}
}

func nativeTestVideo(t *testing.T, first, last string) []byte {
	t.Helper()
	path := filepath.Join(t.TempDir(), "clip.mp4")
	filter := fmt.Sprintf("color=%s:s=256x256:r=1:d=1[a];color=%s:s=256x256:r=1:d=1[b];[a][b]concat=n=2:v=1:a=0", first, last)
	cmd := exec.CommandContext(t.Context(), "ffmpeg", "-v", "error", "-filter_complex_threads", "1",
		"-f", "lavfi", "-i", filter, "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
		"-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-threads", "1", "-c:a", "aac", "-movflags", "+faststart", path)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("create test video with FFmpeg: %v: %s", err, output)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
