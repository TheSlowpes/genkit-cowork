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
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// Discovery reads the GGUF header and tensor directory, never mapping weights.
// See https://github.com/ggml-org/ggml/blob/master/docs/gguf.md.
const (
	ggufHeaderLimit = 128 << 20
	ggufString      = 8
	ggufArray       = 9
)

// GGUF metadata types, in enum order. Zero-width values have variable length.
var ggufWidths = [...]uint64{1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8}

type ggufValue struct {
	kind    uint32
	element uint32 // Element type for arrays.
	text    string
	num     uint64 // Scalar value or array length.
}

type ggufHeader struct {
	values  map[string]ggufValue
	tensors map[string]bool
}

type ggufReader struct {
	*bufio.Reader
	order binary.ByteOrder
}

func (r ggufReader) number(n int) (uint64, error) {
	var buf [8]byte
	if _, err := io.ReadFull(r, buf[:n]); err != nil {
		return 0, err
	}
	switch n {
	case 1:
		return uint64(buf[0]), nil
	case 2:
		return uint64(r.order.Uint16(buf[:])), nil
	case 4:
		return uint64(r.order.Uint32(buf[:])), nil
	default:
		return r.order.Uint64(buf[:]), nil
	}
}

func (r ggufReader) text(limit uint64, keep bool) (string, error) {
	n, err := r.number(8)
	if err != nil {
		return "", err
	}
	if n > limit {
		return "", fmt.Errorf("GGUF string exceeds %d bytes", limit)
	}
	if !keep {
		_, err := r.Discard(int(n))
		return "", err
	}
	buf := make([]byte, n)
	_, err = io.ReadFull(r, buf)
	return string(buf), err
}

func (r ggufReader) value(keep bool) (ggufValue, error) {
	kind, err := r.number(4)
	v := ggufValue{kind: uint32(kind)}
	if err != nil {
		return v, err
	}
	switch {
	case kind >= uint64(len(ggufWidths)):
		return v, fmt.Errorf("unknown GGUF value type %d", kind)
	case kind == ggufString:
		limit := uint64(ggufHeaderLimit)
		if keep {
			limit = 1 << 20
		}
		v.text, err = r.text(limit, keep)
	case kind == ggufArray:
		v.element, v.num, err = r.skipArray()
	default:
		v.num, err = r.number(int(ggufWidths[kind]))
	}
	return v, err
}

func (r ggufReader) skipArray() (uint32, uint64, error) {
	kind, err := r.number(4)
	if err != nil {
		return 0, 0, err
	}
	count, err := r.number(8)
	if err != nil {
		return 0, 0, err
	}
	if kind >= uint64(len(ggufWidths)) || kind == ggufArray || count > 2_000_000 {
		return 0, 0, fmt.Errorf("unsupported GGUF array type %d with %d elements", kind, count)
	}
	if kind != ggufString {
		_, err := r.Discard(int(count * ggufWidths[kind]))
		return uint32(kind), count, err
	}
	for range count {
		if _, err := r.text(ggufHeaderLimit, false); err != nil {
			return 0, 0, err
		}
	}
	return uint32(kind), count, nil
}

func readGGUFHeader(ctx context.Context, path string) (ggufHeader, error) {
	var h ggufHeader
	// Stat first so a symlink to a FIFO cannot block the catalog scan.
	info, err := os.Stat(path)
	if err != nil {
		return h, err
	}
	if !info.Mode().IsRegular() {
		return h, fmt.Errorf("not a regular file")
	}
	f, err := os.Open(path)
	if err != nil {
		return h, err
	}
	defer f.Close()
	limit := &io.LimitedReader{R: f, N: ggufHeaderLimit}
	r := ggufReader{Reader: bufio.NewReaderSize(limit, 32<<10), order: binary.LittleEndian}
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil || string(magic[:]) != "GGUF" {
		return h, fmt.Errorf("missing GGUF magic")
	}
	version, err := r.number(4)
	if version == 3<<24 {
		r.order, version = binary.BigEndian, 3
	}
	if err != nil || (version != 2 && version != 3) {
		return h, fmt.Errorf("unsupported GGUF version %d", version)
	}
	tensors, err := r.number(8)
	if err != nil || tensors > 100_000 {
		return h, fmt.Errorf("invalid GGUF tensor count %d", tensors)
	}
	entries, err := r.number(8)
	if err != nil || entries > 16_384 {
		return h, fmt.Errorf("invalid GGUF metadata entry count %d", entries)
	}
	h.values = make(map[string]ggufValue)
	for range entries {
		if err := ctx.Err(); err != nil {
			return h, err
		}
		key, err := r.text(1024, true)
		if err != nil {
			return h, fmt.Errorf("read GGUF metadata key: %w", err)
		}
		if _, found := h.values[key]; found {
			return h, fmt.Errorf("duplicate GGUF metadata key %q", key)
		}
		keep := key == "general.type" || key == "general.architecture" ||
			key == "tokenizer.ggml.model" || key == "tokenizer.chat_template" ||
			key == "clip.gen.audio.projector_type"
		v, err := r.value(keep)
		if err != nil {
			return h, fmt.Errorf("read GGUF metadata %q: %w", key, err)
		}
		h.values[key] = v
	}
	alignment := uint64(32)
	if v, ok := h.values["general.alignment"]; ok {
		if v.kind != 4 || v.num == 0 || v.num&(v.num-1) != 0 {
			return h, fmt.Errorf("invalid GGUF alignment")
		}
		alignment = v.num
	}
	h.tensors = make(map[string]bool)
	var maxOffset uint64
	for range tensors {
		name, err := r.text(1024, true)
		if err != nil {
			return h, fmt.Errorf("read GGUF tensor name: %w", err)
		}
		if h.tensors[name] {
			return h, fmt.Errorf("duplicate GGUF tensor %q", name)
		}
		h.tensors[name] = true
		dims, err := r.number(4)
		if err != nil || dims == 0 || dims > 4 {
			return h, fmt.Errorf("invalid GGUF tensor dimensions for %q", name)
		}
		if _, err := r.Discard(int(dims*8 + 4)); err != nil {
			return h, fmt.Errorf("read GGUF tensor layout: %w", err)
		}
		offset, err := r.number(8)
		if err != nil || offset%alignment != 0 {
			return h, fmt.Errorf("invalid GGUF tensor offset for %q", name)
		}
		maxOffset = max(maxOffset, offset)
	}
	headerSize := uint64(ggufHeaderLimit - limit.N - int64(r.Buffered()))
	dataStart := (headerSize + alignment - 1) &^ (alignment - 1)
	size := uint64(info.Size())
	if size < dataStart || (tensors > 0 && size-dataStart <= maxOffset) {
		return h, fmt.Errorf("GGUF tensor data is missing or truncated")
	}
	return h, nil
}
