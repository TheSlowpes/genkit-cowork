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
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"iter"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const (
	videoMIME       = "video/mp4"
	maxVideoBytes   = 4 << 20
	maxVideoSeconds = 10
	maxVideoFrames  = 300
	maxVideoEdge    = 1024
	videoSamples    = 4
	videoSampleEdge = 256
	videoTimeout    = 15 * time.Second
)

func validateVideoPair(model map[string]ggufValue, projector ggufHeader) error {
	const verifiedModel = "SmolVLM 256M Instruct"
	if model["general.architecture"].text != "llama" ||
		model["general.basename"].text != "SmolVLM" ||
		model["general.name"].text != verifiedModel ||
		projector.values["general.name"].text != verifiedModel ||
		projector.values["clip.projector_type"].text != "idefics3" {
		return fmt.Errorf("video input requires the SmolVLM 256M Instruct model and matching idefics3 projector")
	}
	return validateVisionPair(model, projector)
}

func decodeVideoPart(payload string, budget int) (MediaInputPart, error) {
	data, err := decodeInlineBytes(payload, min(budget, maxVideoBytes))
	if err != nil {
		return MediaInputPart{}, err
	}
	if err := validateMP4Boxes(data); err != nil {
		return MediaInputPart{}, err
	}
	return MediaInputPart{MIMEType: videoMIME, Data: data}, nil
}

// VideoDecoder replaces the built-in ffprobe/ffmpeg implementation. It
// receives a container-validated MP4 under a 15-second deadline. It must
// verify H.264, duration, dimensions, and frame count before decoding, apply
// the same limits as the built-in decoder, and read only the supplied bytes.
// Return one to four 256x256 samples in display order.
type VideoDecoder interface {
	DecodeVideo(ctx context.Context, mp4 []byte) ([]image.Image, error)
}

type ffmpegVideoDecoder struct{}

func (ffmpegVideoDecoder) DecodeVideo(ctx context.Context, data []byte) ([]image.Image, error) {
	count, err := probeVideo(ctx, data)
	if err != nil {
		return nil, err
	}
	return decodeVideoFrames(ctx, data, count)
}

func (op defaultModelOperator) prepareVideo(ctx context.Context, messages []MediaMessage) error {
	for i, msg := range messages {
		for j := range msg.Parts {
			part := &msg.Parts[j]
			if part.MIMEType != videoMIME {
				continue
			}
			frames, err := decodeVideoSamples(ctx, op.videoDecoder, part.Data)
			if err != nil {
				return fmt.Errorf("yzma: message %d part %d: %w", i, j, err)
			}
			part.VideoFrames = frames
		}
	}
	return nil
}

func decodeVideoSamples(ctx context.Context, decoder VideoDecoder, data []byte) ([]image.Image, error) {
	if decoder == nil {
		decoder = ffmpegVideoDecoder{}
	}
	ctx, cancel := context.WithTimeout(ctx, videoTimeout)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	frames, err := decoder.DecodeVideo(ctx, data)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(frames) < 1 || len(frames) > videoSamples {
		return nil, fmt.Errorf("video decoder must return between 1 and %d samples", videoSamples)
	}
	for _, frame := range frames {
		if frame == nil || frame.Bounds().Size() != image.Pt(videoSampleEdge, videoSampleEdge) {
			return nil, fmt.Errorf("video decoder must return %dx%d samples", videoSampleEdge, videoSampleEdge)
		}
	}
	return frames, nil
}

// validateMP4Boxes requires a complete, non-fragmented container that can
// be decoded from a pipe without seeking back to its media data.
func validateMP4Boxes(data []byte) error {
	ftyp, moov, mdat := false, false, false
	for box, err := range mp4Boxes(data) {
		if err != nil {
			return err
		}
		switch {
		case !ftyp:
			if box.kind != "ftyp" || !isMP4FileType(box.body) {
				return fmt.Errorf("invalid MP4: missing or unrecognized MP4 file type")
			}
			ftyp = true
		case box.kind == "moov":
			if moov {
				return fmt.Errorf("invalid MP4: multiple movie metadata boxes")
			}
			if err := validateMP4MovieDuration(box.body); err != nil {
				return err
			}
			moov = true
		case box.kind == "mdat":
			if !moov {
				return fmt.Errorf("MP4 must be fast-start: metadata must precede media data")
			}
			mdat = true
		case box.kind == "moof":
			return fmt.Errorf("fragmented MP4 is not supported")
		}
	}
	if !mdat {
		return fmt.Errorf("invalid MP4: missing media data")
	}
	return nil
}

func isMP4FileType(body []byte) bool {
	if len(body) < 8 || len(body)%4 != 0 {
		return false
	}
	for i := 0; i < len(body); i += 4 {
		if i == 4 { // This word holds the minor version, not a brand.
			continue
		}
		switch string(body[i : i+4]) {
		case "isom", "iso2", "iso3", "iso4", "iso5", "iso6", "mp41", "mp42", "avc1":
			return true
		}
	}
	return false
}

type mp4Box struct {
	kind string
	body []byte
}

func mp4Boxes(data []byte) iter.Seq2[mp4Box, error] {
	return func(yield func(mp4Box, error) bool) {
		for len(data) > 0 {
			if len(data) < 8 {
				yield(mp4Box{}, fmt.Errorf("invalid MP4: incomplete box header"))
				return
			}
			size := int(binary.BigEndian.Uint32(data))
			if size < 8 || size > len(data) {
				yield(mp4Box{}, fmt.Errorf("invalid MP4: incomplete box or unsupported box size"))
				return
			}
			if !yield(mp4Box{kind: string(data[4:8]), body: data[8:size]}, nil) {
				return
			}
			data = data[size:]
		}
	}
}

// Read the movie header's duration without probing or decoding audio.
func validateMP4MovieDuration(movie []byte) error {
	var scale uint32
	var duration uint64
	for box, err := range mp4Boxes(movie) {
		if err != nil {
			return err
		}
		if box.kind != "mvhd" {
			continue
		}
		body := box.body
		switch {
		case len(body) >= 20 && body[0] == 0:
			scale = binary.BigEndian.Uint32(body[12:16])
			duration = uint64(binary.BigEndian.Uint32(body[16:20]))
		case len(body) >= 32 && body[0] == 1:
			scale = binary.BigEndian.Uint32(body[20:24])
			duration = binary.BigEndian.Uint64(body[24:32])
		}
		break
	}
	if scale == 0 || duration == 0 || duration > maxVideoSeconds*uint64(scale) {
		return fmt.Errorf("MP4 container must have a known duration within (0, %d] seconds", maxVideoSeconds)
	}
	return nil
}

type videoProbe struct {
	Streams []videoTrack  `json:"streams"`
	Packets []videoPacket `json:"packets"`
}

type videoTrack struct {
	Codec    string `json:"codec_name"`
	Width    int    `json:"width"`
	Height   int    `json:"height"`
	TimeBase string `json:"time_base"`
	Duration uint64 `json:"duration_ts"`
	Frames   string `json:"nb_frames"`
}

type videoPacket struct {
	PTS      *uint64 `json:"pts"`
	Duration uint64  `json:"duration"`
}

func (p videoProbe) frameCount() (int, error) {
	if len(p.Streams) != 1 || p.Streams[0].Codec != "h264" {
		return 0, fmt.Errorf("MP4 must have a first video track encoded as H.264")
	}
	s := p.Streams[0]
	if s.Width < 1 || s.Height < 1 || s.Width > maxVideoEdge || s.Height > maxVideoEdge {
		return 0, fmt.Errorf("video dimensions %dx%d exceed the allowed range 1..%d", s.Width, s.Height, maxVideoEdge)
	}
	n, err := strconv.Atoi(s.Frames)
	if err != nil || n < 1 || n > maxVideoFrames || len(p.Packets) != n {
		return 0, fmt.Errorf("MP4 must declare and contain 1..%d source frames", maxVideoFrames)
	}
	a, b, _ := strings.Cut(s.TimeBase, "/")
	num, numErr := strconv.ParseUint(a, 10, 32)
	den, denErr := strconv.ParseUint(b, 10, 32)
	if numErr != nil || denErr != nil || num == 0 || den == 0 {
		return 0, fmt.Errorf("MP4 video track must have a valid time base")
	}
	limit := maxVideoSeconds * den / num
	if s.Duration == 0 || s.Duration > limit {
		return 0, fmt.Errorf("MP4 video track must have a known duration within (0, %d] seconds", maxVideoSeconds)
	}
	for _, pkt := range p.Packets {
		if pkt.PTS == nil || pkt.Duration == 0 || *pkt.PTS > limit || pkt.Duration > limit-*pkt.PTS {
			return 0, fmt.Errorf("MP4 video frame extends outside the %d-second timeline", maxVideoSeconds)
		}
	}
	return n, nil
}

func probeVideo(ctx context.Context, data []byte) (int, error) {
	var probe videoProbe
	err := runVideoCommand(ctx, data, "ffprobe", []string{
		"-v", "error", "-select_streams", "V:0",
		"-read_intervals", "%+#" + strconv.Itoa(maxVideoFrames+1),
		"-show_entries", "stream=codec_name,width,height,time_base,duration_ts,nb_frames:packet=pts,duration",
		"-of", "json",
	}, nil, func(out io.Reader) error {
		const limit = 64 << 10
		metadata, err := io.ReadAll(io.LimitReader(out, limit+1))
		if err != nil {
			return err
		}
		if len(metadata) > limit {
			return fmt.Errorf("video metadata exceeds %d bytes", limit)
		}
		return json.Unmarshal(metadata, &probe)
	})
	if err != nil {
		return 0, err
	}
	return probe.frameCount()
}

func decodeVideoFrames(ctx context.Context, data []byte, count int) ([]image.Image, error) {
	filter := fmt.Sprintf("select='between(t,0,%d)',scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2",
		maxVideoSeconds, videoSampleEdge, videoSampleEdge, videoSampleEdge, videoSampleEdge)
	var frames []image.Image
	err := runVideoCommand(ctx, data, "ffmpeg", []string{
		"-v", "error", "-nostdin", "-xerror", "-err_detect", "explode", "-threads", "1", "-filter_threads", "1", "-an",
	}, []string{
		"-map", "0:V:0", "-an", "-sn", "-dn", "-vf", filter, "-fps_mode", "passthrough",
		"-frames:v", strconv.Itoa(count + 1), "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
	}, func(out io.Reader) error {
		var err error
		frames, err = readVideoFrames(out, count)
		return err
	})
	if err != nil {
		return nil, err
	}
	return frames, nil
}

// readVideoFrames verifies the decoded count and keeps samples spaced evenly by
// frame index, including the first and last. It reuses one RGB frame buffer.
func readVideoFrames(out io.Reader, count int) ([]image.Image, error) {
	samples := min(count, videoSamples)
	frames := make([]image.Image, 0, samples)
	rgb := make([]byte, 3*videoSampleEdge*videoSampleEdge)
	for i := range count {
		if _, err := io.ReadFull(out, rgb); err != nil {
			return nil, fmt.Errorf("expected %d video frames within the duration limit, got %d: %w", count, i, err)
		}
		if len(frames) < samples && i == len(frames)*(count-1)/max(1, samples-1) {
			frame := image.NewRGBA(image.Rect(0, 0, videoSampleEdge, videoSampleEdge))
			for pixel := range videoSampleEdge * videoSampleEdge {
				src, dst := pixel*3, pixel*4
				frame.Pix[dst], frame.Pix[dst+1], frame.Pix[dst+2], frame.Pix[dst+3] = rgb[src], rgb[src+1], rgb[src+2], 255
			}
			frames = append(frames, frame)
		}
	}
	if n, err := io.ReadFull(out, rgb[:1]); n != 0 || err != io.EOF {
		return nil, fmt.Errorf("expected exactly %d video frames within the duration limit", count)
	}
	return frames, nil
}

func runVideoCommand(ctx context.Context, data []byte, name string, before, after []string, consume func(io.Reader) error) error {
	// Only the submitted bytes may be read. In particular, MOV external data
	// references remain disabled and no file or network protocol is permitted.
	args := append(before, "-nofind_stream_info", "-max_alloc", "33554432", "-max_pixels", strconv.Itoa(maxVideoEdge*maxVideoEdge),
		"-protocol_whitelist", "pipe", "-enable_drefs", "0", "-f", "mov", "-i", "pipe:0")
	cmd := exec.CommandContext(ctx, name, append(args, after...)...)
	cmd.Stdin = bytes.NewReader(data)
	cmd.WaitDelay = time.Second
	out, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("%s: open video output pipe: %w", name, err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("%s is required to read inline MP4: %w", name, err)
	}
	readErr := consume(out)
	if readErr != nil {
		_ = out.Close()
		_ = cmd.Process.Kill()
	}
	waitErr := cmd.Wait()
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("%s video decode: %w", name, err)
	}
	if readErr != nil {
		return fmt.Errorf("%s invalid MP4: %w", name, readErr)
	}
	if waitErr != nil {
		return fmt.Errorf("%s could not read MP4: %w", name, waitErr)
	}
	return nil
}
