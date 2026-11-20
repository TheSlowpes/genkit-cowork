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
	"fmt"
	"image"
	_ "image/jpeg" // Register the Go JPEG decoder.
	_ "image/png"  // Register the Go PNG decoder.

	"github.com/hybridgroup/yzma/pkg/mtmd"
)

const (
	maxImageBytes = 4 << 20
	maxImageEdge  = 2048
)

func decodeImagePart(mime, payload string, budget int) (MediaInputPart, error) {
	data, err := decodeInlineBytes(payload, min(budget, maxImageBytes))
	if err != nil {
		return MediaInputPart{}, err
	}
	cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || "image/"+format != mime {
		return MediaInputPart{}, fmt.Errorf("image data is not a valid %s", mime)
	}
	if cfg.Width < 1 || cfg.Height < 1 || cfg.Width > maxImageEdge || cfg.Height > maxImageEdge {
		return MediaInputPart{}, fmt.Errorf("image dimensions %dx%d exceed the allowed range 1..%d", cfg.Width, cfg.Height, maxImageEdge)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return MediaInputPart{}, fmt.Errorf("decode %s: %w", mime, err)
	}
	return MediaInputPart{MIMEType: mime, Data: data, image: img}, nil
}

// imageBitmap copies decoded RGB pixels into a request-owned mtmd bitmap.
// RGB is premultiplied, so transparent areas appear against black.
func imageBitmap(ctx context.Context, img image.Image) (mtmd.Bitmap, error) {
	bounds := img.Bounds()
	rgb := make([]byte, 3*bounds.Dx()*bounds.Dy())
	i := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			rgb[i], rgb[i+1], rgb[i+2] = byte(r>>8), byte(g>>8), byte(b>>8)
			i += 3
		}
	}
	bitmap := mtmd.BitmapInit(uint32(bounds.Dx()), uint32(bounds.Dy()), rgb)
	if bitmap == 0 {
		return 0, fmt.Errorf("allocate image bitmap")
	}
	return bitmap, nil
}
